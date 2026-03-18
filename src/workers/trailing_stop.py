"""Trailing-stop worker — Celery Beat task (every 15 s).

Manages SIM_OPEN trades: fetches open positions, checks latest prices,
applies trailing stop-loss logic, and closes breached positions.

Start the worker + beat scheduler together:
    celery -A src.workers.trailing_stop worker --beat --loglevel=info

Or separately:
    celery -A src.workers.trailing_stop worker --loglevel=info
    celery -A src.workers.trailing_stop beat   --loglevel=info
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Core task
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
