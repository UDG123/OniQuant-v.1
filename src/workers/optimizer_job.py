"""Optuna hyper-parameter optimizer — Celery Beat task (every 24 h).

Queries the ``ml_training_data`` table for the last 1 000 closed trades,
evaluates candidate strategy parameters for all five desks against the
risk-adjusted Sortino ratio, and persists the study state in PostgreSQL so
it survives reboots and supports distributed trial parallelism.

Start the worker + beat scheduler:
    celery -A src.workers.optimizer_job worker --beat --loglevel=info

Or register the beat schedule alongside the trailing-stop worker by
importing ``celery_app`` into a shared config.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import optuna
from celery import Celery
from celery.schedules import crontab
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    Numeric,
    String,
    create_engine,
    text,
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
SYNC_POSTGRES_URL = POSTGRES_URL.replace("+asyncpg", "")

TRADE_LOOKBACK = int(os.getenv("OPTIMIZER_TRADE_LOOKBACK", "1000"))
N_TRIALS = int(os.getenv("OPTIMIZER_N_TRIALS", "60"))
STUDY_NAME = os.getenv("OPTIMIZER_STUDY_NAME", "oniquant_sortino_v1")

ANNUAL_TRADING_PERIODS = 252  # trading days per year

logger = logging.getLogger("oniquant.optimizer")

# Suppress Optuna's default INFO chatter during trial runs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

celery_app = Celery(
    "optimizer_job",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    beat_schedule={
        "nightly-parameter-optimization": {
            "task": "src.workers.optimizer_job.run_optimization",
            "schedule": crontab(hour=3, minute=0),  # 03:00 UTC daily
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
# Database
# ---------------------------------------------------------------------------

_engine = create_engine(SYNC_POSTGRES_URL, pool_pre_ping=True, pool_size=5)
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

Base = declarative_base()


class MLTrainingData(Base):
    """Row-per-closed-trade used as the objective function's input data."""

    __tablename__ = "ml_training_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    desk_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)
    entry_price = Column(Numeric(18, 8), nullable=False)
    close_price = Column(Numeric(18, 8), nullable=False)
    pnl_pct = Column(Float, nullable=False)
    atr_at_entry = Column(Float, nullable=True)
    volume_ratio = Column(Float, nullable=True)
    rsi_at_entry = Column(Float, nullable=True)
    adx_at_entry = Column(Float, nullable=True)
    cvd_zscore = Column(Float, nullable=True)
    lorentzian_score = Column(Float, nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_recent_trades(session: Session) -> list[dict[str, Any]]:
    """Fetch the most recent closed trades from ``ml_training_data``."""
    rows = session.execute(
        text(
            "SELECT desk_id, pnl_pct, atr_at_entry, volume_ratio, "
            "       rsi_at_entry, adx_at_entry, cvd_zscore, lorentzian_score "
            "FROM ml_training_data "
            "ORDER BY closed_at DESC "
            "LIMIT :limit"
        ),
        {"limit": TRADE_LOOKBACK},
    ).fetchall()

    return [dict(r._mapping) for r in rows]


# ---------------------------------------------------------------------------
# Sortino ratio computation
# ---------------------------------------------------------------------------


def _compute_sortino(returns: np.ndarray) -> float:
    """Annualised Sortino ratio from an array of per-trade returns.

    Uses zero as the minimum acceptable return (MAR).  If the downside
    deviation is zero the function returns 0.0 to avoid division errors.
    """
    if len(returns) < 2:
        return 0.0

    mean_return = float(np.mean(returns))
    downside = returns[returns < 0.0]

    if len(downside) < 1:
        return float(mean_return * np.sqrt(ANNUAL_TRADING_PERIODS)) if mean_return > 0 else 0.0

    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0.0:
        return 0.0

    return float((mean_return / downside_std) * np.sqrt(ANNUAL_TRADING_PERIODS))


# ---------------------------------------------------------------------------
# Per-trade filter: simulate whether this trade would have been taken
# under the candidate parameter set.
# ---------------------------------------------------------------------------


def _would_take_trade(trade: dict[str, Any], params: dict[str, Any]) -> bool:
    """Return True if a historical trade passes the candidate filters."""
    desk = trade["desk_id"]

    # --- Desk 1 / 2 / 5: generic CVD z-score gate -----------------------
    if desk in (1, 2, 5):
        cvd = trade.get("cvd_zscore")
        if cvd is not None and abs(cvd) < params["cvd_zscore_threshold"]:
            return False
        return True

    # --- Desk 3: Lorentzian swing filter ---------------------------------
    if desk == 3:
        score = trade.get("lorentzian_score")
        if score is not None and score < params["d3_lorentzian_threshold"]:
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None and not (params["d3_rsi_lower"] < rsi < params["d3_rsi_upper"]):
            return False
        adx = trade.get("adx_at_entry")
        if adx is not None and adx < params["d3_adx_min"]:
            return False
        return True

    # --- Desk 4: Gold breakout filter ------------------------------------
    if desk == 4:
        vol_ratio = trade.get("volume_ratio")
        if vol_ratio is not None and vol_ratio < params["d4_volume_factor"]:
            return False
        atr = trade.get("atr_at_entry")
        if atr is not None and atr <= 0:
            return False
        return True

    return True


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------


def _objective(trial: optuna.Trial, trades: list[dict[str, Any]]) -> float:
    """Optuna objective: maximise the Sortino ratio over filtered trades.

    Each trial samples a full parameter vector covering all five desks.
    The evaluation is entirely self-contained — no shared mutable state.
    """

    params: dict[str, Any] = {
        # Desks 1 / 2 / 5 — CVD z-score gate
        "cvd_zscore_threshold": trial.suggest_float(
            "cvd_zscore_threshold", 0.5, 3.0, step=0.1
        ),
        # Desk 3 — Lorentzian swing
        "d3_neighbours": trial.suggest_int("d3_neighbours", 4, 16),
        "d3_lookback": trial.suggest_int("d3_lookback", 100, 400, step=50),
        "d3_lorentzian_threshold": trial.suggest_float(
            "d3_lorentzian_threshold", 0.4, 0.8, step=0.05
        ),
        "d3_rsi_lower": trial.suggest_float("d3_rsi_lower", 20.0, 40.0, step=5.0),
        "d3_rsi_upper": trial.suggest_float("d3_rsi_upper", 60.0, 80.0, step=5.0),
        "d3_adx_min": trial.suggest_float("d3_adx_min", 15.0, 35.0, step=5.0),
        # Desk 4 — Gold breakout
        "d4_atr_period": trial.suggest_int("d4_atr_period", 7, 21),
        "d4_ema_period": trial.suggest_int("d4_ema_period", 20, 100, step=10),
        "d4_volume_factor": trial.suggest_float(
            "d4_volume_factor", 1.0, 3.0, step=0.25
        ),
        "d4_rr_multiplier": trial.suggest_float(
            "d4_rr_multiplier", 1.0, 4.0, step=0.25
        ),
    }

    # Filter the trade population under the candidate parameters
    accepted_returns: list[float] = [
        t["pnl_pct"]
        for t in trades
        if _would_take_trade(t, params)
    ]

    if len(accepted_returns) < 30:
        # Not enough trades survived the filter — penalise heavily so
        # Optuna steers away from overly restrictive parameter sets.
        return -10.0

    return _compute_sortino(np.array(accepted_returns, dtype=np.float64))


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------


@celery_app.task(
    name="src.workers.optimizer_job.run_optimization",
    bind=True,
    max_retries=1,
    default_retry_delay=300,
)
def run_optimization(self) -> dict[str, Any]:
    """Execute one round of Optuna optimisation over closed trade history.

    The study is persisted in PostgreSQL via the Optuna RDB storage
    backend, so progress survives restarts and multiple workers can run
    trials in parallel.
    """
    db: Session = _SessionLocal()
    try:
        trades = _load_recent_trades(db)
    finally:
        db.close()

    if not trades:
        logger.warning("No trades in ml_training_data — skipping optimisation")
        return {"status": "skipped", "reason": "no_data"}

    logger.info(
        "Starting Optuna study '%s' with %d trades, %d trials",
        STUDY_NAME, len(trades), N_TRIALS,
    )

    storage = optuna.storages.RDBStorage(
        url=SYNC_POSTGRES_URL,
        engine_kwargs={"pool_pre_ping": True, "pool_size": 3},
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: _objective(trial, trades),
        n_trials=N_TRIALS,
    )

    best = study.best_trial
    logger.info(
        "Optimisation complete — best Sortino=%.4f (trial #%d)\n  params=%s",
        best.value,
        best.number,
        best.params,
    )

    return {
        "status": "completed",
        "trades_evaluated": len(trades),
        "n_trials": N_TRIALS,
        "best_trial": best.number,
        "best_sortino": round(best.value, 4),
        "best_params": best.params,
    }
