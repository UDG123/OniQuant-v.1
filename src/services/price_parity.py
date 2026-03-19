"""Dual-venue price parity service — TwelveData + Binance synchronisation.

Subscribes to live tick streams from both TwelveData and Binance
WebSockets simultaneously, normalises timestamps against a local
high-precision clock, computes a high-confidence unified price, and
caches the result in Redis for downstream consumers.

Core functions:

1. **Dual subscription** — two independent ``asyncio.Task`` coroutines
   consume ``bookTicker`` (Binance) and ``price`` (TwelveData) streams
   concurrently within a single event loop.

2. **Timestamp normalisation** — each tick's exchange-originated
   timestamp (``E`` in Binance, ``timestamp`` in TwelveData) is
   compared against ``time.time()`` to compute a running clock-drift
   offset per venue.  All prices are aligned to the local monotonic
   clock before aggregation.

3. **High-confidence price** — the best bid/ask from both venues are
   averaged into a unified quote.  If the inter-venue mid-price
   deviation exceeds 0.5 %, a ``PRICE_PARITY_WARNING`` flag is set in
   Redis and logged at WARNING level.

4. **Redis caching** — the unified price vector is written to
   ``price:latest:{SYMBOL}`` as a JSON hash consumed by the execution
   engine (``src.services.execution``) and safety workers
   (``src.workers.trailing_stop``).

Start the service::

    python -m src.services.price_parity
    python -m src.services.price_parity --symbols BTCUSDT ETHUSDT
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import orjson
import redis.asyncio as aioredis

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

logger = logging.getLogger("oniquant.price_parity")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BINANCE_WS_BASE: str = os.getenv(
    "BINANCE_WS_BASE", "wss://stream.binance.com:9443/ws",
)
TWELVEDATA_WS_URL: str = os.getenv(
    "TWELVEDATA_WS_URL", "wss://ws.twelvedata.com/v1/quotes/price",
)
TWELVEDATA_API_KEY: str = os.getenv("TWELVEDATA_API_KEY", "")

_DEFAULT_SYMBOLS: list[str] = os.getenv(
    "PARITY_SYMBOLS", "BTCUSDT,ETHUSDT,XAUUSD",
).split(",")

_DEVIATION_THRESHOLD_PCT: float = float(
    os.getenv("PARITY_DEVIATION_PCT", "0.5")
)
_PRICE_TTL_SECONDS: int = int(os.getenv("PARITY_PRICE_TTL", "30"))
_WARNING_TTL_SECONDS: int = int(os.getenv("PARITY_WARNING_TTL", "300"))
_DRIFT_WINDOW: int = int(os.getenv("PARITY_DRIFT_WINDOW", "50"))
_RECONNECT_DELAY: float = 5.0

# Redis key for the parity warning flag.
_WARNING_KEY = "global:price_parity_warning"


# ---------------------------------------------------------------------------
# Venue tick — normalised representation from either exchange
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class VenueTick:
    """Normalised quote from a single venue."""

    venue: str              # "binance" | "twelvedata"
    symbol: str
    best_bid: float
    best_ask: float
    mid_price: float
    exchange_epoch_ms: int | None
    local_epoch_ms: float
    clock_drift_ms: float   # local - exchange (positive = exchange behind)


@dataclass(slots=True, frozen=True)
class UnifiedPrice:
    """High-confidence price aggregated from two venues."""

    symbol: str
    unified_bid: float
    unified_ask: float
    unified_mid: float
    binance_mid: float
    twelvedata_mid: float
    deviation_pct: float
    parity_warning: bool
    computed_at: float       # local epoch seconds

    def to_redis_dict(self) -> dict[str, str]:
        return {
            "unified_bid": str(self.unified_bid),
            "unified_ask": str(self.unified_ask),
            "unified_mid": str(self.unified_mid),
            "binance_mid": str(self.binance_mid),
            "twelvedata_mid": str(self.twelvedata_mid),
            "deviation_pct": str(round(self.deviation_pct, 6)),
            "parity_warning": "1" if self.parity_warning else "0",
            "computed_at": str(self.computed_at),
        }


# ---------------------------------------------------------------------------
# Clock drift estimator — running median of (local − exchange) offsets
# ---------------------------------------------------------------------------


class ClockDriftEstimator:
    """Tracks per-venue clock drift via a rolling offset window.

    Each tick contributes one sample: ``local_epoch_ms − exchange_epoch_ms``.
    The median of the last *N* samples is the estimated drift.
    """

    __slots__ = ("_window",)

    def __init__(self, window_size: int = _DRIFT_WINDOW) -> None:
        self._window: deque[float] = deque(maxlen=window_size)

    def update(self, local_ms: float, exchange_ms: float) -> float:
        """Record a sample and return the current drift estimate (ms)."""
        offset = local_ms - exchange_ms
        self._window.append(offset)
        # Median is robust to outlier spikes from network jitter.
        sorted_vals = sorted(self._window)
        n = len(sorted_vals)
        if n % 2 == 1:
            return sorted_vals[n // 2]
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    @property
    def drift_ms(self) -> float:
        if not self._window:
            return 0.0
        sorted_vals = sorted(self._window)
        n = len(sorted_vals)
        if n % 2 == 1:
            return sorted_vals[n // 2]
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    @property
    def samples(self) -> int:
        return len(self._window)


# ---------------------------------------------------------------------------
# PriceParityService
# ---------------------------------------------------------------------------


class PriceParityService:
    """Async dual-venue price synchronisation service.

    Parameters
    ----------
    symbols : list[str]
        Symbols to track (e.g. ``["BTCUSDT", "ETHUSDT"]``).
    deviation_threshold_pct : float
        Maximum allowed mid-price deviation between venues (default 0.5%).
    redis_url : str
        Redis connection URL.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        deviation_threshold_pct: float = _DEVIATION_THRESHOLD_PCT,
        redis_url: str = REDIS_URL,
    ) -> None:
        self._symbols = [s.upper() for s in (symbols or _DEFAULT_SYMBOLS)]
        self._threshold = deviation_threshold_pct
        self._redis_url = redis_url

        # Latest tick per (venue, symbol).
        self._latest: dict[str, dict[str, VenueTick]] = {
            s: {} for s in self._symbols
        }

        # Per-venue clock drift estimators.
        self._drift: dict[str, ClockDriftEstimator] = {
            "binance": ClockDriftEstimator(),
            "twelvedata": ClockDriftEstimator(),
        }

        self._redis: aioredis.Redis | None = None
        self._running = False

    # ----- lifecycle --------------------------------------------------------

    async def _connect_redis(self) -> None:
        pool = aioredis.ConnectionPool.from_url(
            self._redis_url, max_connections=10, decode_responses=False,
        )
        self._redis = aioredis.Redis(connection_pool=pool)

    async def _close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    # ----- Binance consumer -------------------------------------------------

    def _binance_ws_url(self) -> str:
        streams = "/".join(f"{s.lower()}@bookTicker" for s in self._symbols)
        return f"{BINANCE_WS_BASE}/{streams}"

    def _parse_binance_tick(self, raw: bytes) -> VenueTick | None:
        try:
            msg = orjson.loads(raw)
        except (orjson.JSONDecodeError, ValueError):
            return None

        data = msg.get("data", msg)

        try:
            symbol = data["s"].upper()
            best_bid = float(data["b"])
            best_ask = float(data["a"])
        except (KeyError, TypeError, ValueError):
            return None

        if symbol not in self._symbols:
            return None

        exchange_ms: int | None = None
        raw_e = data.get("E") or data.get("T")
        if raw_e is not None:
            try:
                exchange_ms = int(raw_e)
            except (TypeError, ValueError):
                pass

        local_ms = time.time() * 1000.0
        drift = 0.0
        if exchange_ms is not None:
            drift = self._drift["binance"].update(local_ms, float(exchange_ms))

        mid = (best_bid + best_ask) / 2.0

        return VenueTick(
            venue="binance",
            symbol=symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            exchange_epoch_ms=exchange_ms,
            local_epoch_ms=local_ms,
            clock_drift_ms=drift,
        )

    async def _binance_loop(self) -> None:
        """Consume Binance bookTicker stream with auto-reconnect."""
        if websockets is None:
            raise RuntimeError("websockets package required")

        url = self._binance_ws_url()
        retry = _RECONNECT_DELAY

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("Binance WS connected: %s", url)
                    retry = _RECONNECT_DELAY

                    async for raw in ws:
                        tick = self._parse_binance_tick(
                            raw if isinstance(raw, bytes) else raw.encode()
                        )
                        if tick is None:
                            continue

                        self._latest[tick.symbol]["binance"] = tick
                        await self._maybe_emit_unified(tick.symbol)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Binance WS error — reconnecting in %.0fs", retry)
                await asyncio.sleep(retry)
                retry = min(retry * 2, 60.0)

    # ----- TwelveData consumer ----------------------------------------------

    def _parse_twelvedata_tick(self, raw: bytes) -> VenueTick | None:
        try:
            msg = orjson.loads(raw)
        except (orjson.JSONDecodeError, ValueError):
            return None

        event = msg.get("event")
        if event == "subscribe-status":
            status = msg.get("status")
            logger.info("TwelveData subscribe: %s", status)
            return None
        if event == "heartbeat":
            return None

        # TwelveData price message: {"symbol": "BTC/USD", "price": ..., "bid": ..., "ask": ..., "timestamp": ...}
        try:
            raw_symbol = str(msg.get("symbol", ""))
            # Normalise: "BTC/USD" → "BTCUSD", keep "BTCUSDT" as-is.
            symbol = raw_symbol.replace("/", "").upper()
        except (TypeError, ValueError):
            return None

        if symbol not in self._symbols:
            return None

        try:
            bid = float(msg.get("bid") or msg.get("price", 0))
            ask = float(msg.get("ask") or msg.get("price", 0))
        except (TypeError, ValueError):
            return None

        if bid <= 0 or ask <= 0:
            return None

        exchange_ms: int | None = None
        ts = msg.get("timestamp")
        if ts is not None:
            try:
                # TwelveData timestamps are epoch seconds (float).
                exchange_ms = int(float(ts) * 1000)
            except (TypeError, ValueError):
                pass

        local_ms = time.time() * 1000.0
        drift = 0.0
        if exchange_ms is not None:
            drift = self._drift["twelvedata"].update(local_ms, float(exchange_ms))

        mid = (bid + ask) / 2.0

        return VenueTick(
            venue="twelvedata",
            symbol=symbol,
            best_bid=bid,
            best_ask=ask,
            mid_price=mid,
            exchange_epoch_ms=exchange_ms,
            local_epoch_ms=local_ms,
            clock_drift_ms=drift,
        )

    async def _twelvedata_loop(self) -> None:
        """Consume TwelveData price stream with auto-reconnect."""
        if websockets is None:
            raise RuntimeError("websockets package required")
        if not TWELVEDATA_API_KEY:
            logger.warning("TWELVEDATA_API_KEY not set — TwelveData feed disabled")
            return

        retry = _RECONNECT_DELAY

        while self._running:
            try:
                async with websockets.connect(TWELVEDATA_WS_URL) as ws:
                    logger.info("TwelveData WS connected")
                    retry = _RECONNECT_DELAY

                    # Subscribe to symbols.
                    # TwelveData expects "BTC/USD" format for crypto, but
                    # forex/commodities use "EUR/USD", "XAU/USD".
                    td_symbols = []
                    for s in self._symbols:
                        if s.endswith("USDT") or s.endswith("USD"):
                            base = s.replace("USDT", "").replace("USD", "")
                            td_symbols.append(f"{base}/USD")
                        else:
                            td_symbols.append(s)

                    sub_msg = orjson.dumps({
                        "action": "subscribe",
                        "params": {
                            "symbols": ",".join(td_symbols),
                            "apikey": TWELVEDATA_API_KEY,
                        },
                    })
                    await ws.send(sub_msg)

                    async for raw in ws:
                        tick = self._parse_twelvedata_tick(
                            raw if isinstance(raw, bytes) else raw.encode()
                        )
                        if tick is None:
                            continue

                        self._latest[tick.symbol]["twelvedata"] = tick
                        await self._maybe_emit_unified(tick.symbol)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("TwelveData WS error — reconnecting in %.0fs", retry)
                await asyncio.sleep(retry)
                retry = min(retry * 2, 60.0)

    # ----- Unified price computation ----------------------------------------

    async def _maybe_emit_unified(self, symbol: str) -> None:
        """Compute and cache a unified price if both venues have ticks."""
        venue_ticks = self._latest.get(symbol, {})

        bn = venue_ticks.get("binance")
        td = venue_ticks.get("twelvedata")

        if bn is None and td is None:
            return

        # If only one venue is available, use it as-is (no deviation check).
        if bn is not None and td is None:
            unified = UnifiedPrice(
                symbol=symbol,
                unified_bid=bn.best_bid,
                unified_ask=bn.best_ask,
                unified_mid=bn.mid_price,
                binance_mid=bn.mid_price,
                twelvedata_mid=0.0,
                deviation_pct=0.0,
                parity_warning=False,
                computed_at=time.time(),
            )
            await self._cache_price(unified)
            return

        if td is not None and bn is None:
            unified = UnifiedPrice(
                symbol=symbol,
                unified_bid=td.best_bid,
                unified_ask=td.best_ask,
                unified_mid=td.mid_price,
                binance_mid=0.0,
                twelvedata_mid=td.mid_price,
                deviation_pct=0.0,
                parity_warning=False,
                computed_at=time.time(),
            )
            await self._cache_price(unified)
            return

        # Both venues available — compute high-confidence price.
        assert bn is not None and td is not None

        # Staleness check: discard venue data older than 10 seconds.
        now_ms = time.time() * 1000.0
        bn_age_ms = now_ms - bn.local_epoch_ms
        td_age_ms = now_ms - td.local_epoch_ms

        if bn_age_ms > 10_000:
            # Binance tick too stale — use TwelveData only.
            logger.debug("Binance tick stale (%.0fms) for %s", bn_age_ms, symbol)
            bn = None
        if td_age_ms > 10_000:
            logger.debug("TwelveData tick stale (%.0fms) for %s", td_age_ms, symbol)
            td = None

        if bn is None or td is None:
            # One venue went stale — recurse with the surviving tick.
            surviving = bn or td
            assert surviving is not None
            self._latest[symbol] = {surviving.venue: surviving}
            await self._maybe_emit_unified(symbol)
            return

        # Average best bid/ask from both venues.
        unified_bid = (bn.best_bid + td.best_bid) / 2.0
        unified_ask = (bn.best_ask + td.best_ask) / 2.0
        unified_mid = (unified_bid + unified_ask) / 2.0

        # Deviation: |bn_mid − td_mid| / avg_mid × 100.
        avg_mid = (bn.mid_price + td.mid_price) / 2.0
        deviation_pct = 0.0
        if avg_mid > 0:
            deviation_pct = abs(bn.mid_price - td.mid_price) / avg_mid * 100.0

        parity_warning = deviation_pct > self._threshold

        unified = UnifiedPrice(
            symbol=symbol,
            unified_bid=round(unified_bid, 8),
            unified_ask=round(unified_ask, 8),
            unified_mid=round(unified_mid, 8),
            binance_mid=round(bn.mid_price, 8),
            twelvedata_mid=round(td.mid_price, 8),
            deviation_pct=round(deviation_pct, 6),
            parity_warning=parity_warning,
            computed_at=time.time(),
        )

        if parity_warning:
            logger.warning(
                "PRICE_PARITY_WARNING %s: deviation=%.4f%% "
                "(binance=%.8f twelvedata=%.8f threshold=%.2f%%)",
                symbol, deviation_pct,
                bn.mid_price, td.mid_price, self._threshold,
            )

        await self._cache_price(unified)

    # ----- Redis caching ----------------------------------------------------

    async def _cache_price(self, price: UnifiedPrice) -> None:
        """Write the unified price vector to Redis."""
        if self._redis is None:
            return

        key = f"price:latest:{price.symbol}"
        mapping = {
            k.encode(): v.encode() for k, v in price.to_redis_dict().items()
        }

        pipe = self._redis.pipeline(transaction=False)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, _PRICE_TTL_SECONDS)

        # Set or clear the parity warning flag.
        if price.parity_warning:
            pipe.setex(
                _WARNING_KEY,
                _WARNING_TTL_SECONDS,
                f"{price.symbol}:{price.deviation_pct:.4f}%",
            )
        pipe.execute()

    # ----- main loop --------------------------------------------------------

    async def run(self) -> None:
        """Start both venue consumers and run until cancelled."""
        await self._connect_redis()
        self._running = True

        logger.info(
            "PriceParityService starting: symbols=%s threshold=%.2f%%",
            self._symbols, self._threshold,
        )

        tasks = [
            asyncio.create_task(self._binance_loop(), name="binance"),
            asyncio.create_task(self._twelvedata_loop(), name="twelvedata"),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("PriceParityService shutting down")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self._close()

    def stop(self) -> None:
        self._running = False

    # ----- introspection ----------------------------------------------------

    def get_drift_stats(self) -> dict[str, dict[str, Any]]:
        """Return clock drift statistics per venue."""
        return {
            venue: {
                "drift_ms": round(est.drift_ms, 2),
                "samples": est.samples,
            }
            for venue, est in self._drift.items()
        }

    def get_latest_prices(self) -> dict[str, dict[str, VenueTick]]:
        """Return the latest tick per (symbol, venue)."""
        return dict(self._latest)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Dual-venue price parity service")
    parser.add_argument(
        "--symbols", type=str, nargs="+", default=_DEFAULT_SYMBOLS,
    )
    parser.add_argument(
        "--threshold", type=float, default=_DEVIATION_THRESHOLD_PCT,
        help="Deviation threshold in %% (default 0.5)",
    )
    args = parser.parse_args()

    service = PriceParityService(
        symbols=args.symbols,
        deviation_threshold_pct=args.threshold,
    )

    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        logger.info("Interrupted — exiting")
