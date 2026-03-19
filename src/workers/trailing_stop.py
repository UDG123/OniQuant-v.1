"""Trailing-stop worker — Celery Beat task (every 15 s).

Manages SIM_OPEN trades: fetches open positions, checks latest prices,
applies trailing stop-loss logic, and closes breached positions.

Also runs ``check_pending_orders`` on the same 15-second cadence to
drain the Redis pending-signal queue and dispatch triggered signals to
the dynamic limit-order execution engine.

Start the worker + beat scheduler together:
    celery -A src.workers.trailing_stop worker --beat --loglevel=info

Or separately:
    celery -A src.workers.trailing_stop worker --loglevel=info
    celery -A src.workers.trailing_stop beat   --loglevel=info
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import redis as sync_redis
from celery import Celery
from celery.signals import worker_shutting_down
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    Numeric,
    String,
    create_engine,
    update,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.services.execution import (
    ExecutionError,
    OrderBroker,
    execute_dynamic_limit_order,
)
from src.services.redis_manager import RedisStateManager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
)
# Strip the +asyncpg driver suffix so synchronous SQLAlchemy can connect.
SYNC_POSTGRES_URL = POSTGRES_URL.replace("+asyncpg", "")

TRAILING_STOP_PCT = Decimal(os.getenv("TRAILING_STOP_PCT", "0.02"))  # 2 %

logger = logging.getLogger("oniquant.trailing_stop")

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

celery_app = Celery(
    "trailing_stop",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    beat_schedule={
        "trailing-stop-check": {
            "task": "src.workers.trailing_stop.trailing_stop_sweep",
            "schedule": 15.0,
        },
        "pending-orders-check": {
            "task": "src.workers.trailing_stop.check_pending_orders_task",
            "schedule": 15.0,
        },
    },
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# ---------------------------------------------------------------------------
# Database models (lightweight, worker-local)
# ---------------------------------------------------------------------------

Base = declarative_base()


class SimTrade(Base):
    __tablename__ = "sim_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(String(8), nullable=False, default="BUY")
    entry_price = Column(Numeric(18, 8), nullable=False)
    quantity = Column(Numeric(18, 8), nullable=False)
    highest_price = Column(Numeric(18, 8), nullable=False)
    stop_price = Column(Numeric(18, 8), nullable=False)
    status = Column(String(16), nullable=False, default="SIM_OPEN", index=True)
    opened_at = Column(DateTime(timezone=True), nullable=False)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    close_price = Column(Numeric(18, 8), nullable=True)
    desk_id = Column(Integer, nullable=True)


# ---------------------------------------------------------------------------
# Synchronous connection factories (Celery workers are sync by default)
# ---------------------------------------------------------------------------

_engine = create_engine(SYNC_POSTGRES_URL, pool_pre_ping=True, pool_size=5)
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

_redis_client = sync_redis.Redis.from_url(
    REDIS_URL, decode_responses=True, max_connections=10
)

# ---------------------------------------------------------------------------
# In-memory state buffer (flushed on shutdown)
# ---------------------------------------------------------------------------

_pending_updates: list[dict[str, Any]] = []


def _flush_pending_updates() -> None:
    """Write any buffered state changes to the database.

    Called during graceful shutdown to prevent data loss.
    """
    if not _pending_updates:
        return

    logger.info("Flushing %d pending trade updates before shutdown…", len(_pending_updates))
    try:
        db: Session = _SessionLocal()
        try:
            for entry in _pending_updates:
                db.execute(
                    update(SimTrade)
                    .where(SimTrade.id == entry["trade_id"])
                    .values(**entry["values"])
                )
            db.commit()
            logger.info("Flush complete.")
        finally:
            db.close()
    except Exception:
        logger.exception("Failed to flush pending updates on shutdown")
    finally:
        _pending_updates.clear()


# ---------------------------------------------------------------------------
# Graceful shutdown handlers
# ---------------------------------------------------------------------------

_shutting_down = False


def _shutdown_handler(signum: int, _frame: Any) -> None:
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True

    sig_name = signal.Signals(signum).name
    logger.info("Received %s — flushing in-memory state…", sig_name)
    _flush_pending_updates()
    sys.exit(0)


signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)
atexit.register(_flush_pending_updates)


@worker_shutting_down.connect
def _on_worker_shutting_down(sig: str = "", how: str = "", **kwargs: Any) -> None:
    """Celery signal fired just before the worker exits."""
    logger.info("Celery worker_shutting_down (sig=%s, how=%s)", sig, how)
    _flush_pending_updates()


# ---------------------------------------------------------------------------
# Price feed helpers
# ---------------------------------------------------------------------------

_PRICE_KEY_PREFIX = "price:latest:"


def _fetch_latest_prices(symbols: list[str]) -> dict[str, Decimal]:
    """Read the latest price vector from Redis for the requested symbols.

    Expected Redis keys: ``price:latest:<SYMBOL>`` with a numeric string value.
    Symbols without a cached price are silently skipped.
    """
    if not symbols:
        return {}

    keys = [f"{_PRICE_KEY_PREFIX}{s}" for s in symbols]
    raw_values = _redis_client.mget(keys)

    prices: dict[str, Decimal] = {}
    for symbol, raw in zip(symbols, raw_values):
        if raw is not None:
            try:
                prices[symbol] = Decimal(raw)
            except Exception:
                logger.warning("Invalid price value for %s: %r", symbol, raw)
    return prices


def _fetch_latest_price(symbol: str) -> float | None:
    """Return the latest cached price for a single symbol, or ``None``."""
    raw = _redis_client.get(f"{_PRICE_KEY_PREFIX}{symbol}")
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid price value for %s: %r", symbol, raw)
        return None


# ---------------------------------------------------------------------------
# Order-book helper
# ---------------------------------------------------------------------------


def _fetch_order_book(symbol: str) -> dict[str, Any]:
    """Build an order book snapshot from cached Redis price data.

    Constructs best_bid / best_ask from the latest price with a
    synthetic spread.  In production this would query a live L2 feed;
    here we derive it from the cached price to keep the worker
    self-contained.
    """
    price = _fetch_latest_price(symbol)
    if price is None:
        return {}

    # Synthetic 2-bps spread centred on the last traded price.
    half_spread = price * 0.0001
    return {
        "best_bid": round(price - half_spread, 8),
        "best_ask": round(price + half_spread, 8),
        "mid": price,
        "symbol": symbol,
    }


# ---------------------------------------------------------------------------
# Async bridge — run coroutines from synchronous Celery tasks
# ---------------------------------------------------------------------------

_event_loop: asyncio.AbstractEventLoop | None = None


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return a reusable event loop for bridging async calls.

    Celery workers are synchronous, but ``execute_dynamic_limit_order``
    and ``RedisStateManager`` are async.  We maintain a single loop per
    worker process to avoid the overhead of creating / tearing down
    loops on every task invocation.
    """
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
    return _event_loop


def _run_async(coro: Any) -> Any:
    """Execute an async coroutine from synchronous Celery context."""
    loop = _get_or_create_event_loop()
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Pending-orders async core
# ---------------------------------------------------------------------------


async def check_pending_orders(broker: OrderBroker) -> dict[str, Any]:
    """Fetch triggered pending signals from Redis and execute them.

    Steps
    -----
    1. Read the live market price from the Redis price cache.
    2. Query the pending-signal ZSET via
       ``RedisStateManager.get_triggered_signals(current_price)``.
    3. For each triggered signal, build an order-book snapshot and
       dispatch to ``execute_dynamic_limit_order()``.

    Parameters
    ----------
    broker : OrderBroker
        Exchange adapter to forward orders to.

    Returns
    -------
    dict
        Summary with ``triggered``, ``executed``, and ``errors`` counts.
    """
    redis_mgr = RedisStateManager(url=REDIS_URL)
    await redis_mgr.connect()

    triggered_count = 0
    executed_count = 0
    error_count = 0

    try:
        # Collect all unique symbols from the pending queue so we can
        # check each price level.  We scan a broad price range by using
        # a generous current_price to surface everything that is ready.
        #
        # Strategy: read prices for common symbols and check triggers.
        # The ZSET scores are target prices — we query with each
        # symbol's live price to find signals whose target has been hit.

        # Fetch all signals whose target_price <= current market price.
        # We iterate known price keys to cover every symbol that has
        # pending signals.
        all_price_keys: list[str] = []
        cursor: int = 0
        while True:
            cursor, keys = _redis_client.scan(
                cursor=cursor, match=f"{_PRICE_KEY_PREFIX}*", count=200
            )
            all_price_keys.extend(keys)
            if cursor == 0:
                break

        if not all_price_keys:
            logger.debug("No cached prices — skipping pending-order check")
            return {"triggered": 0, "executed": 0, "errors": 0}

        # Determine the highest live price across all symbols so we
        # capture every pending signal whose target has been breached.
        max_price: float = 0.0
        symbol_prices: dict[str, float] = {}
        for key in all_price_keys:
            # key is e.g. "price:latest:XAUUSD"
            sym = key.replace(_PRICE_KEY_PREFIX, "")
            raw = _redis_client.get(key)
            if raw is None:
                continue
            try:
                p = float(raw)
                symbol_prices[sym] = p
                if p > max_price:
                    max_price = p
            except (ValueError, TypeError):
                continue

        if max_price <= 0:
            return {"triggered": 0, "executed": 0, "errors": 0}

        # Pull every signal whose target_price <= max_price.
        triggered_signals: list[dict] = await redis_mgr.get_triggered_signals(
            current_price=max_price,
        )

        if not triggered_signals:
            logger.debug("No pending signals triggered at price %.4f", max_price)
            return {"triggered": 0, "executed": 0, "errors": 0}

        triggered_count = len(triggered_signals)
        logger.info(
            "%d pending signal(s) triggered (max_price=%.4f)",
            triggered_count,
            max_price,
        )

        # Fire executions concurrently for all triggered signals.
        tasks: list[asyncio.Task] = []
        for sig in triggered_signals:
            symbol = sig.get("symbol", "")
            if not symbol:
                error_count += 1
                logger.warning("Triggered signal missing symbol: %s", sig)
                continue

            # Use the symbol-specific live price for the order book.
            live_price = symbol_prices.get(symbol)
            if live_price is None:
                error_count += 1
                logger.warning(
                    "No live price for triggered signal symbol %s", symbol
                )
                continue

            half_spread = live_price * 0.0001
            order_book = {
                "best_bid": round(live_price - half_spread, 8),
                "best_ask": round(live_price + half_spread, 8),
                "mid": live_price,
                "symbol": symbol,
            }

            task = asyncio.create_task(
                _execute_single_signal(sig, order_book, broker)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                error_count += 1
                logger.error("Execution failed for triggered signal: %s", result)
            else:
                executed_count += 1

    finally:
        await redis_mgr.close()

    summary = {
        "triggered": triggered_count,
        "executed": executed_count,
        "errors": error_count,
    }
    logger.info("Pending-order check complete: %s", summary)
    return summary


async def _execute_single_signal(
    sig: dict[str, Any],
    order_book: dict[str, Any],
    broker: OrderBroker,
) -> dict[str, Any]:
    """Execute a single triggered signal through the limit-order engine.

    Wraps ``execute_dynamic_limit_order`` with per-signal error handling
    so one failure does not abort the entire batch.
    """
    symbol = sig.get("symbol", "UNKNOWN")
    signal_id = sig.get("signal_id", "UNKNOWN")

    logger.info(
        "Executing triggered signal %s for %s", signal_id, symbol
    )

    try:
        receipt = await execute_dynamic_limit_order(
            signal=sig,
            current_order_book=order_book,
            broker=broker,
        )
        logger.info(
            "Signal %s executed → %s (filled @ %s, %d chase cycles)",
            signal_id,
            receipt.get("state"),
            receipt.get("filled_price"),
            receipt.get("chase_cycles", 0),
        )
        return receipt
    except ExecutionError as exc:
        logger.error(
            "ExecutionError for signal %s (%s): %s", signal_id, symbol, exc
        )
        raise


# ---------------------------------------------------------------------------
# Stub broker for pending-order dispatch
# ---------------------------------------------------------------------------

# In production, replace ``_default_broker`` with a real exchange adapter
# that satisfies the ``OrderBroker`` protocol (e.g. Binance, Bybit, IBKR).

_default_broker: OrderBroker | None = None


def set_default_broker(broker: OrderBroker) -> None:
    """Register the exchange adapter used by the pending-order worker task."""
    global _default_broker
    _default_broker = broker


def _get_broker() -> OrderBroker:
    """Return the configured broker, or raise if none is registered."""
    if _default_broker is None:
        raise RuntimeError(
            "No OrderBroker registered. Call set_default_broker() at worker "
            "startup or configure an exchange adapter."
        )
    return _default_broker


# ---------------------------------------------------------------------------
# Core tasks
# ---------------------------------------------------------------------------

@celery_app.task(
    name="src.workers.trailing_stop.trailing_stop_sweep",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def trailing_stop_sweep(self) -> dict[str, Any]:
    """Scan all SIM_OPEN trades and apply trailing stop-loss logic.

    For each open trade:
    1. If the latest price exceeds the recorded high, raise the
       ``highest_price`` and recompute the trailing ``stop_price``.
    2. If the latest price has fallen through the ``stop_price``,
       execute a mock sell: set ``status = 'SIM_CLOSED'``.
    """
    if _shutting_down:
        return {"skipped": True, "reason": "worker_shutting_down"}

    db: Session = _SessionLocal()
    try:
        open_trades: list[SimTrade] = (
            db.query(SimTrade)
            .filter(SimTrade.status == "SIM_OPEN")
            .all()
        )

        if not open_trades:
            return {"checked": 0, "closed": 0, "updated": 0}

        # Deduplicated symbol list
        symbols = list({t.symbol for t in open_trades})
        prices = _fetch_latest_prices(symbols)

        closed_count = 0
        updated_count = 0

        for trade in open_trades:
            latest_price = prices.get(trade.symbol)
            if latest_price is None:
                continue

            now = datetime.now(timezone.utc)
            highest = Decimal(str(trade.highest_price))
            stop = Decimal(str(trade.stop_price))

            # --- Price breaches the trailing stop → mock sell ---
            if latest_price <= stop:
                values = {
                    "status": "SIM_CLOSED",
                    "closed_at": now,
                    "close_price": latest_price,
                }
                _pending_updates.append({"trade_id": trade.id, "values": values})

                db.execute(
                    update(SimTrade)
                    .where(SimTrade.id == trade.id)
                    .values(**values)
                )
                # Remove from pending once committed
                _pending_updates.pop()

                closed_count += 1
                logger.info(
                    "CLOSED trade %d (%s) at %s (stop was %s)",
                    trade.id, trade.symbol, latest_price, stop,
                )
                continue

            # --- Price made a new high → ratchet up trailing stop ---
            if latest_price > highest:
                new_highest = latest_price
                new_stop = new_highest * (1 - TRAILING_STOP_PCT)

                values = {
                    "highest_price": new_highest,
                    "stop_price": new_stop,
                }
                _pending_updates.append({"trade_id": trade.id, "values": values})

                db.execute(
                    update(SimTrade)
                    .where(SimTrade.id == trade.id)
                    .values(**values)
                )
                _pending_updates.pop()

                updated_count += 1
                logger.debug(
                    "RATCHET trade %d (%s): high %s → %s, stop → %s",
                    trade.id, trade.symbol, highest, new_highest, new_stop,
                )

        db.commit()

        result = {
            "checked": len(open_trades),
            "closed": closed_count,
            "updated": updated_count,
        }
        logger.info("Sweep complete: %s", result)
        return result

    except Exception as exc:
        db.rollback()
        logger.exception("Trailing-stop sweep failed")
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(
    name="src.workers.trailing_stop.check_pending_orders_task",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def check_pending_orders_task(self) -> dict[str, Any]:
    """Celery task: drain triggered pending signals and execute them.

    Bridges the async ``check_pending_orders()`` coroutine into the
    synchronous Celery worker via a dedicated event loop.

    Runs on the same 15-second Beat cadence as the trailing-stop sweep.
    """
    if _shutting_down:
        return {"skipped": True, "reason": "worker_shutting_down"}

    try:
        broker = _get_broker()
    except RuntimeError:
        logger.warning(
            "No broker configured — skipping pending-order check. "
            "Call set_default_broker() at worker init."
        )
        return {"skipped": True, "reason": "no_broker"}

    try:
        result = _run_async(check_pending_orders(broker))
        return result
    except Exception as exc:
        logger.exception("Pending-order check failed")
        raise self.retry(exc=exc)
