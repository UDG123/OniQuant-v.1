"""Trade execution service — dynamic limit-order engine with chase logic.

Places limit orders strictly at the current best bid (LONG) or best ask
(SHORT) to avoid market-order slippage and adverse selection.  If the
order is not filled within a 30-second window, it is cancelled and
replaced at the freshly quoted best bid/ask.  Once fully filled the
trade record in PostgreSQL is transitioned to ``SIM_OPEN``.

All broker interaction is abstracted behind the ``OrderBroker`` protocol
so the service works against any exchange adapter (live, paper, or
back-test stub).
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import orjson
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    select,
    update,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.services.redis_manager import RedisStateManager

logger = logging.getLogger("oniquant.execution")

# ---------------------------------------------------------------------------
# Module-level emergency halt flag
# ---------------------------------------------------------------------------
_halt_active: bool = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://user:password@localhost:5432/oniquant",
)

_REFRESH_INTERVAL: float = 30.0   # seconds before cancel-replace
_POLL_INTERVAL: float = 1.0       # fill-check cadence within each window
_MAX_CHASE_CYCLES: int = 10       # safety cap: 10 × 30 s = 5 min max

# ---------------------------------------------------------------------------
# Async database engine + session factory
# ---------------------------------------------------------------------------

_engine = create_async_engine(POSTGRES_URL, pool_pre_ping=True, pool_size=5)
_session_factory = async_sessionmaker(
    _engine, class_=AsyncSession, expire_on_commit=False
)


class _Base(DeclarativeBase):
    pass


class TradeState(str, Enum):
    PENDING = "PENDING"
    SIM_OPEN = "SIM_OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class TradeRecord(_Base):
    """Persistent trade lifecycle record."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), nullable=False, unique=True, index=True)
    signal_id = Column(String(128), nullable=False, index=True)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)          # LONG | SHORT
    limit_price = Column(Float, nullable=False)
    filled_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False, default=1.0)
    state = Column(
        String(16), nullable=False, default=TradeState.PENDING.value
    )
    desk_id = Column(Integer, nullable=True)
    broker_order_id = Column(String(128), nullable=True)
    chase_cycles = Column(Integer, nullable=False, default=0)
    raw_signal = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Broker protocol — exchange adapter interface
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OrderStatus:
    """Snapshot of a broker order's current state."""
    order_id: str
    is_filled: bool
    filled_price: float | None = None
    filled_qty: float | None = None
    remaining_qty: float | None = None
    raw: dict = field(default_factory=dict)


@runtime_checkable
class OrderBroker(Protocol):
    """Minimal async interface that any exchange adapter must satisfy."""

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
    ) -> str:
        """Submit a limit order and return the broker order ID."""
        ...

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Return True if successfully cancelled."""
        ...

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Poll current fill status for *order_id*."""
        ...

    async def get_order_book(self, symbol: str) -> dict:
        """Return a fresh ``{"best_bid": float, "best_ask": float, ...}``."""
        ...


# ---------------------------------------------------------------------------
# Core execution logic
# ---------------------------------------------------------------------------


class ExecutionError(Exception):
    """Raised when the execution engine encounters an unrecoverable error."""


def _extract_limit_price(side: str, order_book: dict) -> float:
    """Pick best bid for LONG, best ask for SHORT.

    Raises ``ExecutionError`` if the required level is missing or invalid.
    """
    if side == "LONG":
        price = order_book.get("best_bid")
    elif side == "SHORT":
        price = order_book.get("best_ask")
    else:
        raise ExecutionError(f"Unknown side: {side!r} (expected LONG or SHORT)")

    if price is None or not isinstance(price, (int, float)) or price <= 0:
        raise ExecutionError(
            f"Invalid {('best_bid' if side == 'LONG' else 'best_ask')} "
            f"in order book: {price!r}"
        )
    return float(price)


def _derive_side(action: str) -> str:
    """Map webhook action → execution side."""
    mapping = {"BUY": "LONG", "SELL": "SHORT", "LONG": "LONG", "SHORT": "SHORT"}
    side = mapping.get(action.upper())
    if side is None:
        raise ExecutionError(f"Cannot derive side from action: {action!r}")
    return side


async def _transition_to_sim_open(
    db: AsyncSession,
    trade_id: str,
    filled_price: float,
    chase_cycles: int,
) -> None:
    """Atomically mark a trade as SIM_OPEN in PostgreSQL."""
    now = datetime.now(timezone.utc)
    stmt = (
        update(TradeRecord)
        .where(TradeRecord.trade_id == trade_id)
        .values(
            state=TradeState.SIM_OPEN.value,
            filled_price=filled_price,
            filled_at=now,
            updated_at=now,
            chase_cycles=chase_cycles,
        )
    )
    await db.execute(stmt)
    await db.commit()
    logger.info(
        "Trade %s → SIM_OPEN @ %.8f after %d chase cycle(s)",
        trade_id,
        filled_price,
        chase_cycles,
    )


async def _mark_expired(db: AsyncSession, trade_id: str) -> None:
    """Mark the trade as EXPIRED after exhausting chase cycles."""
    now = datetime.now(timezone.utc)
    stmt = (
        update(TradeRecord)
        .where(TradeRecord.trade_id == trade_id)
        .values(state=TradeState.EXPIRED.value, updated_at=now)
    )
    await db.execute(stmt)
    await db.commit()
    logger.warning("Trade %s → EXPIRED (max chase cycles reached)", trade_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_dynamic_limit_order(
    signal: dict[str, Any],
    current_order_book: dict[str, Any],
    *,
    broker: OrderBroker,
    db: AsyncSession | None = None,
    refresh_interval: float = _REFRESH_INTERVAL,
    max_chase_cycles: int = _MAX_CHASE_CYCLES,
) -> dict[str, Any]:
    """Place a limit order at the best bid/ask and chase until filled.

    Parameters
    ----------
    signal : dict
        Webhook / hydrated signal payload.  Must contain at minimum
        ``signal_id``, ``symbol``, and ``action`` (BUY | SELL).
        Optionally: ``quantity`` (default 1.0), ``desk_id``.
    current_order_book : dict
        Must contain ``best_bid`` and ``best_ask`` as positive floats.
    broker : OrderBroker
        Exchange adapter implementing the ``OrderBroker`` protocol.
    db : AsyncSession | None
        Async SQLAlchemy session.  A new one is created if ``None``.
    refresh_interval : float
        Seconds to wait before cancel-replace (default 30).
    max_chase_cycles : int
        Maximum cancel-replace iterations before giving up (default 10).

    Returns
    -------
    dict
        Execution receipt with keys: ``trade_id``, ``state``,
        ``filled_price``, ``chase_cycles``, ``broker_order_id``.

    Raises
    ------
    ExecutionError
        On missing fields, invalid order-book data, or broker failures.
    """
    # --- Emergency halt gate -----------------------------------------------
    if _halt_active:
        raise ExecutionError(
            "Emergency halt is active — all new executions are blocked"
        )

    # --- Validate inputs --------------------------------------------------
    signal_id: str = signal.get("signal_id", "")
    symbol: str = signal.get("symbol", "")
    action: str = signal.get("action", "")
    quantity: float = float(signal.get("quantity", 1.0))
    desk_id: int | None = signal.get("desk_id")

    if not signal_id or not symbol or not action:
        raise ExecutionError(
            "Signal must contain non-empty signal_id, symbol, and action"
        )

    side: str = _derive_side(action)
    limit_price: float = _extract_limit_price(side, current_order_book)
    trade_id: str = f"trade-{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc)

    # --- Acquire DB session -----------------------------------------------
    own_session = db is None
    if own_session:
        db = _session_factory()

    try:
        # --- Persist initial PENDING record -------------------------------
        record = TradeRecord(
            trade_id=trade_id,
            signal_id=signal_id,
            symbol=symbol,
            side=side,
            limit_price=limit_price,
            quantity=quantity,
            state=TradeState.PENDING.value,
            desk_id=desk_id,
            chase_cycles=0,
            created_at=now,
            updated_at=now,
        )
        db.add(record)
        await db.commit()

        # --- Place initial limit order ------------------------------------
        broker_order_id: str = await broker.place_limit_order(
            symbol=symbol,
            side=side,
            price=limit_price,
            quantity=quantity,
        )
        logger.info(
            "Limit order placed: %s %s %s @ %.8f (order=%s)",
            trade_id, side, symbol, limit_price, broker_order_id,
        )

        # --- Chase loop: poll → cancel → replace -------------------------
        chase_cycle = 0

        while chase_cycle < max_chase_cycles:
            filled_price = await _wait_for_fill(
                broker, broker_order_id, refresh_interval
            )

            if filled_price is not None:
                await _transition_to_sim_open(
                    db, trade_id, filled_price, chase_cycle
                )
                return {
                    "trade_id": trade_id,
                    "state": TradeState.SIM_OPEN.value,
                    "filled_price": filled_price,
                    "chase_cycles": chase_cycle,
                    "broker_order_id": broker_order_id,
                }

            # --- Not filled — cancel and replace --------------------------
            chase_cycle += 1
            cancelled = await broker.cancel_order(broker_order_id)
            if not cancelled:
                # Order may have filled between the last poll and cancel.
                status = await broker.get_order_status(broker_order_id)
                if status.is_filled:
                    fp = status.filled_price or limit_price
                    await _transition_to_sim_open(
                        db, trade_id, fp, chase_cycle
                    )
                    return {
                        "trade_id": trade_id,
                        "state": TradeState.SIM_OPEN.value,
                        "filled_price": fp,
                        "chase_cycles": chase_cycle,
                        "broker_order_id": broker_order_id,
                    }

            # Fetch fresh order book and re-price.
            fresh_book: dict = await broker.get_order_book(symbol)
            limit_price = _extract_limit_price(side, fresh_book)

            broker_order_id = await broker.place_limit_order(
                symbol=symbol,
                side=side,
                price=limit_price,
                quantity=quantity,
            )
            logger.info(
                "Chase cycle %d: replaced order for %s @ %.8f (order=%s)",
                chase_cycle, trade_id, limit_price, broker_order_id,
            )

        # --- Exhausted all chase cycles -----------------------------------
        await broker.cancel_order(broker_order_id)
        await _mark_expired(db, trade_id)

        return {
            "trade_id": trade_id,
            "state": TradeState.EXPIRED.value,
            "filled_price": None,
            "chase_cycles": chase_cycle,
            "broker_order_id": broker_order_id,
        }

    finally:
        if own_session:
            await db.close()


async def _wait_for_fill(
    broker: OrderBroker,
    order_id: str,
    timeout: float,
) -> float | None:
    """Poll the broker for a fill within *timeout* seconds.

    Returns the filled price on success, ``None`` if the timeout elapses
    without a complete fill.
    """
    elapsed = 0.0
    while elapsed < timeout:
        if _halt_active:
            return None
        status: OrderStatus = await broker.get_order_status(order_id)
        if status.is_filled:
            return status.filled_price
        await asyncio.sleep(_POLL_INTERVAL)
        elapsed += _POLL_INTERVAL
    return None


# ---------------------------------------------------------------------------
# Emergency halt — cancel all pending limit orders
# ---------------------------------------------------------------------------


async def cancel_all_pending_orders(
    broker: OrderBroker,
    db: AsyncSession | None = None,
) -> dict[str, Any]:
    """Cancel every trade in PENDING state and mark it CANCELLED.

    Called by the emergency-halt Pub/Sub listener to immediately abort
    all in-flight limit orders.

    Returns
    -------
    dict
        Summary with ``cancelled`` count and ``errors`` count.
    """
    own_session = db is None
    if own_session:
        db = _session_factory()

    cancelled = 0
    errors = 0

    try:
        result = await db.execute(
            select(TradeRecord).where(
                TradeRecord.state == TradeState.PENDING.value,
            )
        )
        pending_trades: list[TradeRecord] = list(result.scalars().all())

        if not pending_trades:
            logger.info("Emergency halt: no PENDING orders to cancel")
            return {"cancelled": 0, "errors": 0}

        logger.warning(
            "Emergency halt: cancelling %d PENDING orders", len(pending_trades)
        )

        now = datetime.now(timezone.utc)
        for trade in pending_trades:
            try:
                if trade.broker_order_id:
                    await broker.cancel_order(trade.broker_order_id)
                await db.execute(
                    update(TradeRecord)
                    .where(TradeRecord.id == trade.id)
                    .values(
                        state=TradeState.CANCELLED.value,
                        updated_at=now,
                    )
                )
                cancelled += 1
                logger.info(
                    "Emergency halt: cancelled order %s (broker=%s)",
                    trade.trade_id,
                    trade.broker_order_id,
                )
            except Exception:
                errors += 1
                logger.exception(
                    "Emergency halt: failed to cancel order %s",
                    trade.trade_id,
                )

        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("Emergency halt: cancel_all_pending_orders failed")
        raise
    finally:
        if own_session:
            await db.close()

    summary = {"cancelled": cancelled, "errors": errors}
    logger.warning("Emergency halt cancel complete: %s", summary)
    return summary


async def listen_for_emergency_halt(
    redis_url: str | None = None,
    broker: OrderBroker | None = None,
) -> None:
    """Long-running coroutine that listens for EMERGENCY_HALT events.

    When a halt message arrives on the ``oniquant:emergency_halt``
    Pub/Sub channel, this listener:

    1. Sets the module-level ``_halt_active`` flag to block new orders.
    2. Calls ``cancel_all_pending_orders()`` to abort in-flight orders.

    When a ``HALT_LIFTED`` message arrives, the flag is cleared.

    Intended to be launched as a background task during the FastAPI
    lifespan or similar startup hook::

        asyncio.create_task(listen_for_emergency_halt(broker=my_broker))
    """
    global _halt_active

    redis_mgr = RedisStateManager(url=redis_url)
    await redis_mgr.connect()

    # Sync flag with existing Redis state on startup.
    _halt_active = await redis_mgr.is_emergency_halt_active()
    if _halt_active:
        logger.warning(
            "Emergency halt already active on startup — blocking new orders"
        )

    try:
        pubsub = await redis_mgr.subscribe_emergency_halt()
        logger.info("Emergency halt listener started on Pub/Sub channel")

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            try:
                event = orjson.loads(message["data"])
            except (orjson.JSONDecodeError, ValueError):
                logger.warning(
                    "Malformed emergency halt message: %r", message["data"]
                )
                continue

            event_type = event.get("event")

            if event_type == "EMERGENCY_HALT":
                _halt_active = True
                logger.warning(
                    "EMERGENCY HALT received — reason: %s", event.get("reason")
                )

                if broker is not None:
                    try:
                        await cancel_all_pending_orders(broker)
                    except Exception:
                        logger.exception(
                            "Failed to cancel pending orders during halt"
                        )

            elif event_type == "HALT_LIFTED":
                _halt_active = False
                logger.info("Emergency halt lifted — resuming normal operation")

    finally:
        await redis_mgr.close()
