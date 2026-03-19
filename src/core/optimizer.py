"""Autonomous optimization engine — RecursiveOptimizer.

Connects to PostgreSQL to fetch ``ml_training_data`` (historical closed
trades), constructs a dynamic Optuna search space covering all five
trading desks, and maximises the annualised Sortino ratio through
iterative trial evaluation.

Unlike the Celery-bound ``optimizer_job`` worker, this module exposes a
reusable ``RecursiveOptimizer`` class that can be embedded in any async
or sync context — FastAPI lifespan, Jupyter notebooks, CLI scripts, or
the existing Celery beat schedule.

Usage::

    optimizer = RecursiveOptimizer(postgres_url="postgresql://…")
    result = optimizer.run(n_trials=120)
    print(result.best_params, result.best_sortino)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
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
# Configuration defaults
# ---------------------------------------------------------------------------

_DEFAULT_POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
).replace("+asyncpg", "")

_DEFAULT_STUDY_NAME: str = os.getenv(
    "OPTIMIZER_STUDY_NAME", "oniquant_recursive_sortino_v2"
)
_DEFAULT_TRADE_LOOKBACK: int = int(os.getenv("OPTIMIZER_TRADE_LOOKBACK", "1000"))
_DEFAULT_N_TRIALS: int = int(os.getenv("OPTIMIZER_N_TRIALS", "80"))
_MIN_ACCEPTED_TRADES: int = 30
_ANNUAL_TRADING_PERIODS: int = 252
_INSUFFICIENT_TRADES_PENALTY: float = -10.0

logger = logging.getLogger("oniquant.core.optimizer")

# Suppress Optuna's default INFO chatter during trial runs.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Lightweight ORM model (worker-local copy to stay decoupled from async Base)
# ---------------------------------------------------------------------------

_Base = declarative_base()


class _MLTrainingData(_Base):
    """Read-only projection of ``ml_training_data`` for the optimizer."""

    __tablename__ = "ml_training_data"

    id = Column(Integer, primary_key=True)
    desk_id = Column(Integer, nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)
    entry_price = Column(Numeric(18, 8), nullable=False)
    close_price = Column(Numeric(18, 8), nullable=False)
    pnl_pct = Column(Float, nullable=False)
    atr_at_entry = Column(Float, nullable=True)
    volume_ratio = Column(Float, nullable=True)
    rsi_at_entry = Column(Float, nullable=True)
    adx_at_entry = Column(Float, nullable=True)
    cci_at_entry = Column(Float, nullable=True)
    ema_distance = Column(Float, nullable=True)
    cvd_zscore = Column(Float, nullable=True)
    lorentzian_score = Column(Float, nullable=True)
    consensus_score = Column(Float, nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OptimizationResult:
    """Immutable snapshot of a completed optimization run."""

    study_name: str
    n_trials_completed: int
    trades_evaluated: int
    best_trial_number: int
    best_sortino: float
    best_params: dict[str, Any]
    param_importances: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------


def _compute_sortino(returns: np.ndarray) -> float:
    """Annualised Sortino ratio (MAR = 0).

    Returns 0.0 when there are fewer than 2 trades or the downside
    deviation is zero.
    """
    if len(returns) < 2:
        return 0.0

    mean_return = float(np.mean(returns))
    downside = returns[returns < 0.0]

    if len(downside) < 1:
        # All trades profitable — reward proportionally.
        return (
            float(mean_return * np.sqrt(_ANNUAL_TRADING_PERIODS))
            if mean_return > 0
            else 0.0
        )

    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0.0:
        return 0.0

    return float((mean_return / downside_std) * np.sqrt(_ANNUAL_TRADING_PERIODS))


# ---------------------------------------------------------------------------
# Trade filter — would this historical trade pass candidate params?
# ---------------------------------------------------------------------------


def _would_take_trade(trade: dict[str, Any], params: dict[str, Any]) -> bool:
    """Simulate whether *trade* would have been accepted under *params*."""
    desk = trade["desk_id"]

    # --- Desk 1: OFI scalping ------------------------------------------------
    if desk == 1:
        cvd = trade.get("cvd_zscore")
        if cvd is not None and abs(cvd) < params["d1_ofi_threshold"]:
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None and not (params["d1_rsi_lower"] < rsi < params["d1_rsi_upper"]):
            return False
        return True

    # --- Desk 2: Kalman FX ---------------------------------------------------
    if desk == 2:
        cvd = trade.get("cvd_zscore")
        if cvd is not None and abs(cvd) < params["d2_kalman_z_entry"]:
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None and not (params["d2_rsi_oversold"] < rsi < params["d2_rsi_overbought"]):
            return False
        return True

    # --- Desk 3: Lorentzian swing --------------------------------------------
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
        cci = trade.get("cci_at_entry")
        if cci is not None and abs(cci) < params["d3_cci_min_abs"]:
            return False
        return True

    # --- Desk 4: Gold breakout -----------------------------------------------
    if desk == 4:
        vol_ratio = trade.get("volume_ratio")
        if vol_ratio is not None and vol_ratio < params["d4_volume_factor"]:
            return False
        atr = trade.get("atr_at_entry")
        if atr is not None and atr <= 0:
            return False
        ema_dist = trade.get("ema_distance")
        if ema_dist is not None and abs(ema_dist) < params["d4_ema_min_distance"]:
            return False
        return True

    # --- Desk 5: CVD crypto --------------------------------------------------
    if desk == 5:
        cvd = trade.get("cvd_zscore")
        if cvd is not None and abs(cvd) < params["d5_cvd_threshold"]:
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None and not (params["d5_rsi_lower"] < rsi < params["d5_rsi_upper"]):
            return False
        return True

    return True


# ---------------------------------------------------------------------------
# Search space definition (define-by-run)
# ---------------------------------------------------------------------------


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Construct the full search space using Optuna's define-by-run API.

    Every parameter is sampled via ``trial.suggest_float`` or
    ``trial.suggest_int`` so the search space is dynamically built per
    trial — no static grid required.
    """
    params: dict[str, Any] = {}

    # ── Desk 1 — OFI scalp ──────────────────────────────────────────
    params["d1_ofi_window"] = trial.suggest_int("d1_ofi_window", 8, 30)
    params["d1_ofi_threshold"] = trial.suggest_float(
        "d1_ofi_threshold", 0.30, 0.85, step=0.05
    )
    params["d1_price_flat_pct"] = trial.suggest_float(
        "d1_price_flat_pct", 0.05, 0.40, step=0.05
    )
    params["d1_atr_stop_mult"] = trial.suggest_float(
        "d1_atr_stop_mult", 1.0, 3.0, step=0.25
    )
    params["d1_rsi_lower"] = trial.suggest_float(
        "d1_rsi_lower", 20.0, 40.0, step=5.0
    )
    params["d1_rsi_upper"] = trial.suggest_float(
        "d1_rsi_upper", 60.0, 80.0, step=5.0
    )

    # ── Desk 2 — Kalman FX ──────────────────────────────────────────
    params["d2_kalman_z_entry"] = trial.suggest_float(
        "d2_kalman_z_entry", 1.0, 3.5, step=0.1
    )
    params["d2_kalman_process_var"] = trial.suggest_float(
        "d2_kalman_process_var", 1e-6, 1e-3, log=True
    )
    params["d2_kalman_measurement_var"] = trial.suggest_float(
        "d2_kalman_measurement_var", 1e-4, 1e-1, log=True
    )
    params["d2_rsi_oversold"] = trial.suggest_float(
        "d2_rsi_oversold", 20.0, 40.0, step=5.0
    )
    params["d2_rsi_overbought"] = trial.suggest_float(
        "d2_rsi_overbought", 60.0, 80.0, step=5.0
    )
    params["d2_atr_stop_mult"] = trial.suggest_float(
        "d2_atr_stop_mult", 1.0, 3.0, step=0.25
    )

    # ── Desk 3 — Lorentzian swing ───────────────────────────────────
    params["d3_neighbours"] = trial.suggest_int("d3_neighbours", 3, 20)
    params["d3_lookback"] = trial.suggest_int("d3_lookback", 80, 500, step=20)
    params["d3_lorentzian_threshold"] = trial.suggest_float(
        "d3_lorentzian_threshold", 0.35, 0.85, step=0.05
    )
    params["d3_rsi_lower"] = trial.suggest_float(
        "d3_rsi_lower", 15.0, 40.0, step=5.0
    )
    params["d3_rsi_upper"] = trial.suggest_float(
        "d3_rsi_upper", 60.0, 85.0, step=5.0
    )
    params["d3_adx_min"] = trial.suggest_float(
        "d3_adx_min", 10.0, 40.0, step=5.0
    )
    params["d3_cci_min_abs"] = trial.suggest_float(
        "d3_cci_min_abs", 50.0, 200.0, step=10.0
    )
    params["d3_ema_period"] = trial.suggest_int("d3_ema_period", 20, 100, step=10)

    # ── Desk 4 — Gold breakout ──────────────────────────────────────
    params["d4_atr_period"] = trial.suggest_int("d4_atr_period", 7, 28)
    params["d4_ema_period"] = trial.suggest_int("d4_ema_period", 20, 120, step=10)
    params["d4_volume_factor"] = trial.suggest_float(
        "d4_volume_factor", 1.0, 3.5, step=0.25
    )
    params["d4_rr_multiplier"] = trial.suggest_float(
        "d4_rr_multiplier", 1.0, 5.0, step=0.25
    )
    params["d4_ema_min_distance"] = trial.suggest_float(
        "d4_ema_min_distance", 0.0, 2.0, step=0.1
    )

    # ── Desk 5 — CVD crypto ─────────────────────────────────────────
    params["d5_cvd_window"] = trial.suggest_int("d5_cvd_window", 10, 40, step=2)
    params["d5_cvd_threshold"] = trial.suggest_float(
        "d5_cvd_threshold", 0.5, 3.5, step=0.1
    )
    params["d5_atr_regime_mult"] = trial.suggest_float(
        "d5_atr_regime_mult", 1.05, 1.80, step=0.05
    )
    params["d5_atr_stop_mult"] = trial.suggest_float(
        "d5_atr_stop_mult", 1.0, 4.0, step=0.25
    )
    params["d5_rsi_lower"] = trial.suggest_float(
        "d5_rsi_lower", 20.0, 40.0, step=5.0
    )
    params["d5_rsi_upper"] = trial.suggest_float(
        "d5_rsi_upper", 60.0, 80.0, step=5.0
    )

    # ── Cross-desk: consensus score gate ─────────────────────────────
    params["min_consensus_score"] = trial.suggest_float(
        "min_consensus_score", 0.3, 0.9, step=0.05
    )

    return params


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _objective(
    trial: optuna.Trial,
    trades: list[dict[str, Any]],
) -> float:
    """Optuna objective: maximise Sortino over parameter-filtered trades.

    The search space is constructed dynamically on every call via the
    define-by-run API — no static parameter grid is required.
    """
    params = _suggest_params(trial)

    # Apply cross-desk consensus gate first.
    min_consensus = params["min_consensus_score"]

    accepted_returns: list[float] = []
    for t in trades:
        cs = t.get("consensus_score")
        if cs is not None and cs < min_consensus:
            continue
        if _would_take_trade(t, params):
            accepted_returns.append(t["pnl_pct"])

    if len(accepted_returns) < _MIN_ACCEPTED_TRADES:
        return _INSUFFICIENT_TRADES_PENALTY

    returns_arr = np.array(accepted_returns, dtype=np.float64)
    sortino = _compute_sortino(returns_arr)

    # Report intermediate metrics for dashboard monitoring.
    trial.set_user_attr("n_accepted_trades", len(accepted_returns))
    trial.set_user_attr("mean_return", float(np.mean(returns_arr)))
    trial.set_user_attr("win_rate", float(np.mean(returns_arr > 0)))

    return sortino


# ---------------------------------------------------------------------------
# RecursiveOptimizer
# ---------------------------------------------------------------------------


class RecursiveOptimizer:
    """Autonomous parameter optimization engine backed by Optuna.

    Fetches ``ml_training_data`` from PostgreSQL, constructs a dynamic
    search space covering all five trading desks, and iteratively
    maximises the annualised Sortino ratio.

    Parameters
    ----------
    postgres_url : str | None
        Synchronous SQLAlchemy connection URL.  Falls back to the
        ``POSTGRES_URL`` environment variable (with ``+asyncpg``
        stripped).
    study_name : str
        Optuna study name.  Set ``load_if_exists=True`` to resume.
    trade_lookback : int
        Number of most-recent closed trades to evaluate.
    sampler : optuna.samplers.BaseSampler | None
        Custom Optuna sampler.  Defaults to TPE with multivariate
        enabled for correlated parameter modelling.
    pruner : optuna.pruners.BasePruner | None
        Custom Optuna pruner.  Defaults to ``MedianPruner``.
    """

    def __init__(
        self,
        postgres_url: str | None = None,
        study_name: str = _DEFAULT_STUDY_NAME,
        trade_lookback: int = _DEFAULT_TRADE_LOOKBACK,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
    ) -> None:
        self._pg_url: str = (postgres_url or _DEFAULT_POSTGRES_URL).replace(
            "+asyncpg", ""
        )
        self._study_name = study_name
        self._trade_lookback = trade_lookback

        self._sampler = sampler or optuna.samplers.TPESampler(
            multivariate=True,
            seed=42,
        )
        self._pruner = pruner or optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=0,
        )

        self._engine = create_engine(
            self._pg_url, pool_pre_ping=True, pool_size=5
        )
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False
        )

        self._study: optuna.Study | None = None
        self._trades: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_trades(self) -> list[dict[str, Any]]:
        """Fetch the most recent closed trades from ``ml_training_data``."""
        session: Session = self._session_factory()
        try:
            rows = session.execute(
                text(
                    "SELECT desk_id, pnl_pct, atr_at_entry, volume_ratio, "
                    "       rsi_at_entry, adx_at_entry, cci_at_entry, "
                    "       ema_distance, cvd_zscore, lorentzian_score, "
                    "       consensus_score "
                    "FROM ml_training_data "
                    "ORDER BY closed_at DESC "
                    "LIMIT :limit"
                ),
                {"limit": self._trade_lookback},
            ).fetchall()
            return [dict(r._mapping) for r in rows]
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Study lifecycle
    # ------------------------------------------------------------------

    def _get_or_create_study(self) -> optuna.Study:
        """Create or load the Optuna study with PostgreSQL storage."""
        storage = optuna.storages.RDBStorage(
            url=self._pg_url,
            engine_kwargs={"pool_pre_ping": True, "pool_size": 3},
        )
        return optuna.create_study(
            study_name=self._study_name,
            storage=storage,
            direction="maximize",
            sampler=self._sampler,
            pruner=self._pruner,
            load_if_exists=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        n_trials: int = _DEFAULT_N_TRIALS,
        timeout: float | None = None,
    ) -> OptimizationResult:
        """Execute an optimization cycle.

        Parameters
        ----------
        n_trials : int
            Number of Optuna trials to run (default 80).
        timeout : float | None
            Optional wall-clock limit in seconds.

        Returns
        -------
        OptimizationResult
            Snapshot of the best trial and its parameters.
        """
        self._trades = self._load_trades()
        if not self._trades:
            logger.warning("No trades in ml_training_data — returning empty result")
            return OptimizationResult(
                study_name=self._study_name,
                n_trials_completed=0,
                trades_evaluated=0,
                best_trial_number=-1,
                best_sortino=0.0,
                best_params={},
            )

        logger.info(
            "RecursiveOptimizer starting: study=%s, trades=%d, trials=%d",
            self._study_name,
            len(self._trades),
            n_trials,
        )

        self._study = self._get_or_create_study()
        trades = self._trades

        self._study.optimize(
            lambda trial: _objective(trial, trades),
            n_trials=n_trials,
            timeout=timeout,
        )

        best = self._study.best_trial

        # Compute parameter importances (requires ≥ 2 completed trials).
        importances: dict[str, float] = {}
        completed = [
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed) >= 2:
            try:
                importances = optuna.importance.get_param_importances(
                    self._study
                )
            except Exception:
                logger.debug("Could not compute param importances", exc_info=True)

        result = OptimizationResult(
            study_name=self._study_name,
            n_trials_completed=len(completed),
            trades_evaluated=len(self._trades),
            best_trial_number=best.number,
            best_sortino=round(best.value, 6),
            best_params=dict(best.params),
            param_importances=importances,
        )

        logger.info(
            "Optimization complete — best Sortino=%.4f (trial #%d)\n"
            "  params=%s",
            result.best_sortino,
            result.best_trial_number,
            result.best_params,
        )

        return result

    @property
    def study(self) -> optuna.Study | None:
        """Access the underlying Optuna study (``None`` before ``run()``)."""
        return self._study

    @property
    def best_params(self) -> dict[str, Any]:
        """Shortcut to the best trial's parameter dict."""
        if self._study is None:
            return {}
        return dict(self._study.best_params)

    def get_desk_params(self, desk_id: int) -> dict[str, Any]:
        """Extract optimized parameters for a single desk.

        Filters the best parameter set to keys prefixed with the desk
        identifier (e.g. ``d2_`` for desk 2), strips the prefix, and
        returns a clean dict suitable for passing to the desk's strategy
        constructor.
        """
        prefix = f"d{desk_id}_"
        return {
            k[len(prefix):]: v
            for k, v in self.best_params.items()
            if k.startswith(prefix)
        }
