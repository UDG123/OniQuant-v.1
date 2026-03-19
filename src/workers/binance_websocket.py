"""Binance WebSocket consumer — real-time Micro-Price & Order Book Imbalance.

Connects to the Binance ``bookTicker`` WebSocket stream, computes two
microstructure metrics on every tick, and injects them into the Desk 1
Redis state for downstream signal hydration:

1. **Micro-Price** — volume-weighted fair value estimate::

       S = (P_ask · V_bid + P_bid · V_ask) / (V_bid + V_ask)

2. **Order Book Imbalance (OBI) Z-Score** — standardised imbalance over
   a 100-tick rolling window::

       OBI_raw  = (V_bid - V_ask) / (V_bid + V_ask)     ∈ [-1, +1]
       OBI_z    = (OBI_raw - μ_100) / σ_100

Dispatch logic:

* **OBI Z > +2.0** → fire "Aggressive Entry" (BUY) signal for Desk 1.
* **OBI Z < −2.0** → fire "Aggressive Exit" (SELL) signal for Desk 1.

Signals flow through the existing webhook pipeline (hydration → ClaudeCTO
→ execution) so all risk gates remain active.

Start the consumer::

    python -m src.workers.binance_websocket
    python -m src.workers.binance_websocket --symbol BTCUSDT --z-threshold 2.5

Or import and run programmatically::

    from src.workers.binance_websocket import BinanceBookTickerConsumer
    consumer = BinanceBookTickerConsumer(symbols=["BTCUSDT", "ETHUSDT"])
    asyncio.run(consumer.run())
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import orjson
import redis.asyncio as aioredis

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BINANCE_WS_BASE: str = os.getenv(
    "BINANCE_WS_BASE", "wss://stream.binance.com:9443/ws",
)
WEBHOOK_URL: str = os.getenv(
    "ONIQUANT_WEBHOOK_URL", "http://localhost:8000/api/v1/tradingview-alert",
)

_DEFAULT_SYMBOL: str = os.getenv("BINANCE_WS_SYMBOL", "BTCUSDT")
_OBI_WINDOW: int = int(os.getenv("OBI_WINDOW", "100"))
_Z_THRESHOLD: float = float(os.getenv("OBI_Z_THRESHOLD", "2.0"))
_COOLDOWN_SECONDS: float = float(os.getenv("OBI_SIGNAL_COOLDOWN", "60.0"))
_RECONNECT_DELAY: float = 5.0
_DESK_ID: int = 1

logger = logging.getLogger("oniquant.binance_ws")

# ---------------------------------------------------------------------------
# Rolling statistics buffer (lock-free, O(1) per tick)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RollingOBIBuffer:
    """Fixed-size circular buffer that maintains running mean and variance.

    Uses Welford's online algorithm adapted for a sliding window so each
    tick update is O(1) without recomputing over the full window.
    """

    window: int = _OBI_WINDOW
    _values: deque = field(default_factory=lambda: deque(maxlen=_OBI_WINDOW))
    _sum: float = 0.0
    _sum_sq: float = 0.0

    def push(self, obi_raw: float) -> None:
        """Append a new OBI observation, evicting the oldest if full."""
        if len(self._values) == self.window:
            evicted = self._values[0]
            self._sum -= evicted
            self._sum_sq -= evicted * evicted

        self._values.append(obi_raw)
        self._sum += obi_raw
        self._sum_sq += obi_raw * obi_raw

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        n = self.count
        return self._sum / n if n > 0 else 0.0

    @property
    def std(self) -> float:
        n = self.count
        if n < 2:
            return 0.0
        variance = (self._sum_sq / n) - (self._sum / n) ** 2
        # Guard against negative variance from float rounding.
        return float(np.sqrt(max(variance, 0.0)))

    def z_score(self, obi_raw: float) -> float:
        """Compute the z-score of *obi_raw* against the rolling window."""
        s = self.std
        if s == 0.0 or self.count < self.window:
            return 0.0
        return (obi_raw - self.mean) / s

    @property
    def is_warm(self) -> bool:
        """True once the buffer has accumulated a full window of ticks."""
        return self.count >= self.window


# ---------------------------------------------------------------------------
# Micro-price computation
# ---------------------------------------------------------------------------


def compute_micro_price(
    best_bid: float,
    best_ask: float,
    bid_qty: float,
    ask_qty: float,
) -> float:
    """Volume-weighted micro-price estimating true fair value.

    S = (P_ask · V_bid + P_bid · V_ask) / (V_bid + V_ask)

    Falls back to the mid-price when total volume is zero.
    """
    total_qty = bid_qty + ask_qty
    if total_qty <= 0.0:
        return (best_bid + best_ask) / 2.0
    return (best_ask * bid_qty + best_bid * ask_qty) / total_qty


def compute_obi_raw(bid_qty: float, ask_qty: float) -> float:
    """Raw Order Book Imbalance ∈ [-1, +1].

    OBI = (V_bid - V_ask) / (V_bid + V_ask)
    """
    total = bid_qty + ask_qty
    if total <= 0.0:
        return 0.0
    return (bid_qty - ask_qty) / total


# ---------------------------------------------------------------------------
# Tick dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class BookTick:
    """Parsed Binance bookTicker message."""

    symbol: str
    best_bid: float
    best_ask: float
    bid_qty: float
    ask_qty: float
    micro_price: float
    obi_raw: float
    obi_zscore: float
    received_at: float


# ---------------------------------------------------------------------------
# WebSocket consumer
# ---------------------------------------------------------------------------


class BinanceBookTickerConsumer:
    """Async Binance ``bookTicker`` WebSocket consumer with OBI dispatch.

    Parameters
    ----------
    symbols : list[str]
        Symbols to subscribe to (default ``["BTCUSDT"]``).
    obi_window : int
        Rolling window size for z-score normalisation (default 100).
    z_threshold : float
        Absolute OBI z-score threshold for signal dispatch (default 2.0).
    cooldown : float
        Minimum seconds between successive signals for the same symbol.
    redis_url : str
        Redis connection URL for desk state writes.
    webhook_url : str
        Internal webhook URL to inject signals into the pipeline.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        obi_window: int = _OBI_WINDOW,
        z_threshold: float = _Z_THRESHOLD,
        cooldown: float = _COOLDOWN_SECONDS,
        redis_url: str = REDIS_URL,
        webhook_url: str = WEBHOOK_URL,
    ) -> None:
        self._symbols = [s.lower() for s in (symbols or [_DEFAULT_SYMBOL])]
        self._z_threshold = z_threshold
        self._cooldown = cooldown
        self._redis_url = redis_url
        self._webhook_url = webhook_url

        # Per-symbol rolling buffers.
        self._buffers: dict[str, RollingOBIBuffer] = {
            s: RollingOBIBuffer(window=obi_window) for s in self._symbols
        }

        # Cooldown tracker: symbol → last signal epoch.
        self._last_signal: dict[str, float] = {s: 0.0 for s in self._symbols}

        # Latest tick per symbol (for Redis state).
        self._latest_tick: dict[str, BookTick | None] = {
            s: None for s in self._symbols
        }

        self._redis: aioredis.Redis | None = None
        self._http: Any = None
        self._running = False

    # ----- lifecycle --------------------------------------------------------

    async def _connect_redis(self) -> None:
        pool = aioredis.ConnectionPool.from_url(
            self._redis_url, max_connections=10, decode_responses=False,
        )
        self._redis = aioredis.Redis(connection_pool=pool)

    async def _get_http(self) -> Any:
        if httpx is None:
            raise RuntimeError("httpx is required for signal dispatch")
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=10.0)
        return self._http

    async def _close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()
        if self._redis:
            await self._redis.aclose()

    # ----- stream URL -------------------------------------------------------

    def _build_ws_url(self) -> str:
        """Construct the combined stream URL for all symbols."""
        streams = "/".join(f"{s}@bookTicker" for s in self._symbols)
        return f"{BINANCE_WS_BASE}/{streams}"

    # ----- message processing -----------------------------------------------

    def _parse_tick(self, raw: bytes) -> BookTick | None:
        """Parse a raw Binance bookTicker JSON message into a BookTick."""
        try:
            msg = orjson.loads(raw)
        except (orjson.JSONDecodeError, ValueError):
            return None

        # Combined stream wraps in {"stream": …, "data": {…}}.
        data = msg.get("data", msg)

        try:
            symbol = data["s"].lower()
            best_bid = float(data["b"])
            best_ask = float(data["a"])
            bid_qty = float(data["B"])
            ask_qty = float(data["A"])
        except (KeyError, TypeError, ValueError):
            return None

        if symbol not in self._buffers:
            return None

        micro = compute_micro_price(best_bid, best_ask, bid_qty, ask_qty)
        obi_raw = compute_obi_raw(bid_qty, ask_qty)

        buf = self._buffers[symbol]
        buf.push(obi_raw)
        obi_z = buf.z_score(obi_raw)

        return BookTick(
            symbol=symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            micro_price=round(micro, 8),
            obi_raw=round(obi_raw, 6),
            obi_zscore=round(obi_z, 4),
            received_at=time.time(),
        )

    # ----- Redis desk state update ------------------------------------------

    async def _update_desk_state(self, tick: BookTick) -> None:
        """Append microstructure metrics to the Desk 1 Redis state hash.

        These fields are available to ``hydrate_signal()`` and downstream
        strategy evaluators:

        * ``micro_price`` — volume-weighted fair value
        * ``obi_raw`` — raw order book imbalance [-1, +1]
        * ``obi_zscore`` — standardised OBI over the rolling window
        * ``best_bid``, ``best_ask``, ``bid_qty``, ``ask_qty``
        * ``obi_updated_at`` — epoch timestamp of the last tick
        """
        if self._redis is None:
            return

        key = f"desk:{_DESK_ID}:state"
        mapping: dict[bytes, bytes] = {
            b"micro_price": orjson.dumps(tick.micro_price),
            b"obi_raw": orjson.dumps(tick.obi_raw),
            b"obi_zscore": orjson.dumps(tick.obi_zscore),
            b"best_bid": orjson.dumps(tick.best_bid),
            b"best_ask": orjson.dumps(tick.best_ask),
            b"bid_qty": orjson.dumps(tick.bid_qty),
            b"ask_qty": orjson.dumps(tick.ask_qty),
            b"obi_updated_at": orjson.dumps(tick.received_at),
        }
        await self._redis.hset(key, mapping=mapping)  # type: ignore[arg-type]

    # ----- signal dispatch --------------------------------------------------

    async def _dispatch_signal(
        self, tick: BookTick, action: str, reason: str,
    ) -> None:
        """POST an aggressive signal to the internal webhook endpoint.

        The signal enters the standard pipeline (dedup → volatility lock
        → hydration → ClaudeCTO → execution) so all risk gates remain
        active.
        """
        signal_id = f"obi-{tick.symbol}-{uuid.uuid4().hex[:12]}"
        payload = {
            "signal_id": signal_id,
            "symbol": tick.symbol.upper(),
            "action": action,
            "desk_id": _DESK_ID,
            "price": tick.micro_price,
            "message": (
                f"{reason} | obi_z={tick.obi_zscore:.2f} "
                f"micro={tick.micro_price:.8f} "
                f"bid_qty={tick.bid_qty:.4f} ask_qty={tick.ask_qty:.4f}"
            ),
        }

        try:
            client = await self._get_http()
            resp = await client.post(
                self._webhook_url,
                content=orjson.dumps(payload),
                headers={"content-type": "application/json"},
            )
            logger.info(
                "OBI signal dispatched: %s %s %s (z=%.2f) → HTTP %d",
                action, tick.symbol.upper(), reason,
                tick.obi_zscore, resp.status_code,
            )
        except Exception:
            logger.exception(
                "Failed to dispatch OBI signal %s for %s",
                action, tick.symbol.upper(),
            )

    def _check_cooldown(self, symbol: str) -> bool:
        """Return True if enough time has elapsed since the last signal."""
        now = time.time()
        elapsed = now - self._last_signal.get(symbol, 0.0)
        return elapsed >= self._cooldown

    async def _evaluate_dispatch(self, tick: BookTick) -> None:
        """Check OBI z-score thresholds and dispatch if warranted."""
        buf = self._buffers.get(tick.symbol)
        if buf is None or not buf.is_warm:
            return

        if not self._check_cooldown(tick.symbol):
            return

        if tick.obi_zscore > self._z_threshold:
            await self._dispatch_signal(tick, "BUY", "Aggressive Entry")
            self._last_signal[tick.symbol] = time.time()

        elif tick.obi_zscore < -self._z_threshold:
            await self._dispatch_signal(tick, "SELL", "Aggressive Exit")
            self._last_signal[tick.symbol] = time.time()

    # ----- main loop --------------------------------------------------------

    async def run(self) -> None:
        """Connect to Binance and process ticks indefinitely.

        Automatically reconnects with exponential back-off on
        disconnection.  Gracefully handles ``KeyboardInterrupt`` and
        ``asyncio.CancelledError``.
        """
        if websockets is None:
            raise RuntimeError(
                "The 'websockets' package is required — "
                "install via: pip install websockets"
            )

        await self._connect_redis()
        self._running = True
        url = self._build_ws_url()

        logger.info(
            "Binance WS consumer starting: symbols=%s window=%d z_threshold=%.1f",
            [s.upper() for s in self._symbols],
            _OBI_WINDOW,
            self._z_threshold,
        )

        retry_delay = _RECONNECT_DELAY

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("Connected to %s", url)
                    retry_delay = _RECONNECT_DELAY  # Reset on success.

                    async for raw_message in ws:
                        tick = self._parse_tick(
                            raw_message
                            if isinstance(raw_message, bytes)
                            else raw_message.encode()
                        )
                        if tick is None:
                            continue

                        self._latest_tick[tick.symbol] = tick

                        # Fire-and-forget Redis update + dispatch eval.
                        await self._update_desk_state(tick)
                        await self._evaluate_dispatch(tick)

            except asyncio.CancelledError:
                logger.info("Consumer cancelled — shutting down")
                break
            except Exception:
                logger.exception(
                    "WebSocket disconnected — reconnecting in %.0fs",
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60.0)

        await self._close()
        logger.info("Binance WS consumer stopped")

    def stop(self) -> None:
        """Signal the consumer to shut down gracefully."""
        self._running = False

    # ----- introspection ----------------------------------------------------

    def get_latest_tick(self, symbol: str) -> BookTick | None:
        """Return the most recent parsed tick for *symbol*."""
        return self._latest_tick.get(symbol.lower())

    def get_buffer_stats(self, symbol: str) -> dict[str, Any]:
        """Return rolling buffer statistics for *symbol*."""
        buf = self._buffers.get(symbol.lower())
        if buf is None:
            return {}
        return {
            "count": buf.count,
            "is_warm": buf.is_warm,
            "mean": round(buf.mean, 6),
            "std": round(buf.std, 6),
            "window": buf.window,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Binance bookTicker consumer with Micro-Price & OBI",
    )
    parser.add_argument(
        "--symbol", type=str, nargs="+", default=[_DEFAULT_SYMBOL],
        help="Symbol(s) to subscribe to (default: BTCUSDT)",
    )
    parser.add_argument(
        "--window", type=int, default=_OBI_WINDOW,
        help="OBI rolling window size (default: 100)",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=_Z_THRESHOLD,
        help="OBI Z-score threshold for signal dispatch (default: 2.0)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=_COOLDOWN_SECONDS,
        help="Minimum seconds between signals per symbol (default: 60)",
    )
    args = parser.parse_args()

    consumer = BinanceBookTickerConsumer(
        symbols=args.symbol,
        obi_window=args.window,
        z_threshold=args.z_threshold,
        cooldown=args.cooldown,
    )

    try:
        asyncio.run(consumer.run())
    except KeyboardInterrupt:
        logger.info("Interrupted — exiting")
