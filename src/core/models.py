"""Canonical async SQLAlchemy models for OniQuant.

All tables are defined here as the single source of truth.  Worker-local
lightweight copies (e.g. in ``trailing_stop.py``) remain for backward
compatibility but should converge to import from this module over time.

Usage with asyncpg::

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    engine = create_async_engine("postgresql+asyncpg://…")
    async with async_sessionmaker(engine)() as session:
        ...
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all OniQuant models."""


# ---------------------------------------------------------------------------
# TradeLog — full lifecycle of every simulated position
# ---------------------------------------------------------------------------


class TradeLog(Base):
    """Tracks trades from SIM_OPEN through SIM_CLOSED.

    The ``mfe_mae_ratio`` (Maximum Favourable / Maximum Adverse Excursion)
    quantifies how efficiently the trade captured its best opportunity
    relative to its worst drawdown.
    """

    __tablename__ = "trade_log"
    __table_args__ = (
        Index("ix_trade_log_desk_status", "desk_id", "status"),
        Index("ix_trade_log_closed_at", "closed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    desk_id: Mapped[int] = mapped_column(Integer, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False, default="BUY")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="SIM_OPEN", index=True,
    )

    # Price columns — Numeric(18,8) matches the precision used across the codebase
    entry_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)

    # Trailing-stop bookkeeping
    highest_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    lowest_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    stop_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)

    # MFE / MAE analysis
    mfe: Mapped[float | None] = mapped_column(
        Numeric(18, 8), nullable=True, comment="Maximum Favourable Excursion",
    )
    mae: Mapped[float | None] = mapped_column(
        Numeric(18, 8), nullable=True, comment="Maximum Adverse Excursion",
    )
    mfe_mae_ratio: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="MFE / MAE — trade quality score",
    )

    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )


# ---------------------------------------------------------------------------
# VetoLog — signals dropped by the Redis Hot-Path risk gates
# ---------------------------------------------------------------------------


class VetoLog(Base):
    """Records every signal rejected before it could become a trade.

    Common veto reasons include ``volatility_lock``, ``max_exposure``,
    ``cooldown_active``, and ``duplicate_signal``.
    """

    __tablename__ = "veto_log"
    __table_args__ = (
        Index("ix_veto_log_vetoed_at", "vetoed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    desk_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    reason: Mapped[str] = mapped_column(
        String(64), nullable=False, default="volatility_lock",
    )
    raw_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    vetoed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )


# ---------------------------------------------------------------------------
# MLTrainingData — feature vectors consumed by the Optuna optimizer
# ---------------------------------------------------------------------------


class MLTrainingData(Base):
    """One row per closed trade with the feature snapshot captured at entry.

    The Optuna optimizer (``src.workers.optimizer_job``) queries this table
    for the last N trades to evaluate candidate parameter sets.  The
    ``consensus_score`` is the AI ensemble's final agreement metric at the
    time the signal was generated.
    """

    __tablename__ = "ml_training_data"
    __table_args__ = (
        Index("ix_ml_training_desk_closed", "desk_id", "closed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    desk_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)

    # Price snapshot
    entry_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    close_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=False)

    # Technical indicators captured at entry
    atr_at_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    rsi_at_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    adx_at_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    cci_at_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_distance: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Price distance from EMA as pct",
    )

    # Composite / model scores
    cvd_zscore: Mapped[float | None] = mapped_column(Float, nullable=True)
    lorentzian_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    consensus_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="AI ensemble agreement metric (0-1)",
    )

    # Timestamps
    closed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
