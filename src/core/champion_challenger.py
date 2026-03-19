"""Champion–Challenger orchestrator — Celery Chord-based parameter promotion.

Parallelises two out-of-sample backtests via a Celery ``chord``:

1. **Champion**: evaluates the current production parameter set (best
   trial from the existing Optuna study) over the most recent 1 000
   trades from ``ml_training_data``.
2. **Challenger**: runs a fresh Optuna mutation cycle on the *in-sample*
   partition (first 800 trades) to discover a candidate parameter set,
   then evaluates it on the *out-of-sample* holdout (last 200 trades).

A chord callback performs an **Independent-Samples Welch's T-test** on
block-bootstrapped Sortino ratios.  The Challenger is promoted to
Champion (hot-swapped into Redis) **only** when the Sortino improvement
is statistically significant at *p < 0.05*.

Run the orchestrator manually::

    from src.core.champion_challenger import trigger_champion_challenger
    trigger_champion_challenger.delay()

Or schedule it via Celery Beat alongside the nightly optimizer::

    "champion-challenger": {
        "task": "src.core.champion_challenger.trigger_champion_challenger",
        "schedule": crontab(hour=4, minute=0),
    }
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import optuna
from celery import Celery, chord
from scipy import stats
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
)
SYNC_POSTGRES_URL = POSTGRES_URL.replace("+asyncpg", "")

# Trade dataset sizes.
_TOTAL_LOOKBACK: int = int(os.getenv("CC_TOTAL_LOOKBACK", "1000"))
_OOS_SIZE: int = int(os.getenv("CC_OOS_SIZE", "200"))

# Optuna mutation budget for the Challenger.
_CHALLENGER_N_TRIALS: int = int(os.getenv("CC_CHALLENGER_TRIALS", "60"))
_CHALLENGER_STUDY_NAME: str = os.getenv(
    "CC_CHALLENGER_STUDY", "oniquant_challenger",
)
_CHAMPION_STUDY_NAME: str = os.getenv(
    "CC_CHAMPION_STUDY", "oniquant_evolution",
)

# Statistical significance threshold for promotion.
_P_VALUE_THRESHOLD: float = float(os.getenv("CC_P_VALUE_THRESHOLD", "0.05"))

# Block size for bootstrapped Sortino samples (trades per block).
_BLOCK_SIZE: int = int(os.getenv("CC_BLOCK_SIZE", "20"))

# Minimum Sortino improvement (absolute) to even consider promotion.
_MIN_SORTINO_IMPROVEMENT: float = float(
    os.getenv("CC_MIN_SORTINO_IMPROVEMENT", "0.05"),
)

_MIN_ACCEPTED_TRADES: int = 30
_ANNUAL_TRADING_PERIODS: int = 252
_INSUFFICIENT_TRADES_PENALTY: float = -10.0

logger = logging.getLogger("oniquant.champion_challenger")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

celery_app = Celery(
    "champion_challenger",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_track_started=True,
    result_expires=86_400,
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_engine = create_engine(SYNC_POSTGRES_URL, pool_pre_ping=True, pool_size=5)
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_TRADE_COLUMNS = (
    "desk_id, pnl_pct, atr_at_entry, volume_ratio, "
    "rsi_at_entry, adx_at_entry, cci_at_entry, "
    "ema_distance, cvd_zscore, lorentzian_score, consensus_score"
)


def _load_recent_trades(limit: int = _TOTAL_LOOKBACK) -> list[dict[str, Any]]:
    """Fetch the most recent *limit* closed trades from ``ml_training_data``.

    Returns trades in chronological order (oldest first) so that slicing
    ``[:N]`` gives the in-sample partition and ``[N:]`` the out-of-sample
    holdout.
    """
    db: Session = _SessionLocal()
    try:
        rows = db.execute(
            text(
                f"SELECT {_TRADE_COLUMNS} "
                "FROM ml_training_data "
                "ORDER BY closed_at DESC "
                "LIMIT :limit"
            ),
            {"limit": limit},
        ).fetchall()
        # Reverse so index 0 = oldest.
        trades = [dict(r._mapping) for r in reversed(rows)]
        return trades
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------


def _compute_sortino(returns: np.ndarray) -> float:
    """Annualised Sortino ratio (MAR = 0)."""
    if len(returns) < 2:
        return 0.0

    mean_return = float(np.mean(returns))
    downside = returns[returns < 0.0]

    if len(downside) < 1:
        return (
            float(mean_return * np.sqrt(_ANNUAL_TRADING_PERIODS))
            if mean_return > 0
            else 0.0
        )

    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0.0:
        return 0.0

    return float((mean_return / downside_std) * np.sqrt(_ANNUAL_TRADING_PERIODS))


def _compute_block_sortinos(returns: np.ndarray, block_size: int) -> np.ndarray:
    """Split *returns* into non-overlapping blocks and compute Sortino per block.

    This produces independent observations suitable for a T-test,
    avoiding the single-point-estimate problem of computing one
    aggregate Sortino over the entire return series.

    Parameters
    ----------
    returns : np.ndarray
        Per-trade return series.
    block_size : int
        Number of trades per block.

    Returns
    -------
    np.ndarray
        Array of block-level Sortino ratios.
    """
    n_blocks = len(returns) // block_size
    if n_blocks < 2:
        # Not enough data for meaningful blocks — return the single aggregate.
        return np.array([_compute_sortino(returns)])

    sortinos: list[float] = []
    for i in range(n_blocks):
        block = returns[i * block_size : (i + 1) * block_size]
        sortinos.append(_compute_sortino(block))

    return np.array(sortinos, dtype=np.float64)


# ---------------------------------------------------------------------------
# Trade filter (mirrors src/core/recursive_engine._would_take_trade)
# ---------------------------------------------------------------------------


def _would_take_trade(trade: dict[str, Any], params: dict[str, Any]) -> bool:
    """Simulate whether *trade* would have been accepted under *params*."""
    desk = trade["desk_id"]

    if desk == 1:
        cvd = trade.get("cvd_zscore")
        if cvd is not None and abs(cvd) < params.get("d1_ofi_threshold", 0):
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None and not (
            params.get("d1_rsi_lower", 0) < rsi < params.get("d1_rsi_upper", 100)
        ):
            return False
        return True

    if desk == 2:
        cvd = trade.get("cvd_zscore")
        if cvd is not None and abs(cvd) < params.get("d2_kalman_z_entry", 0):
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None and not (
            params.get("d2_rsi_oversold", 0) < rsi < params.get("d2_rsi_overbought", 100)
        ):
            return False
        return True

    if desk == 3:
        score = trade.get("lorentzian_score")
        if score is not None and score < params.get("d3_lorentzian_threshold", 0):
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None and not (
            params.get("d3_rsi_lower", 0) < rsi < params.get("d3_rsi_upper", 100)
        ):
            return False
        adx = trade.get("adx_at_entry")
        if adx is not None and adx < params.get("d3_adx_min", 0):
            return False
        cci = trade.get("cci_at_entry")
        if cci is not None and abs(cci) < params.get("d3_cci_min_abs", 0):
            return False
        return True

    if desk == 4:
        vol_ratio = trade.get("volume_ratio")
        if vol_ratio is not None and vol_ratio < params.get("d4_volume_factor", 0):
            return False
        atr = trade.get("atr_at_entry")
        if atr is not None and atr <= 0:
            return False
        ema_dist = trade.get("ema_distance")
        if ema_dist is not None and abs(ema_dist) < params.get("d4_ema_min_distance", 0):
            return False
        return True

    if desk == 5:
        cvd = trade.get("cvd_zscore")
        if cvd is not None and abs(cvd) < params.get("d5_cvd_threshold", 0):
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None and not (
            params.get("d5_rsi_lower", 0) < rsi < params.get("d5_rsi_upper", 100)
        ):
            return False
        return True

    return True


def _filter_returns(
    trades: list[dict[str, Any]],
    params: dict[str, Any],
) -> np.ndarray:
    """Apply *params* to *trades* and return the accepted pnl_pct series."""
    min_consensus = params.get("min_consensus_score", 0.0)

    accepted: list[float] = []
    for t in trades:
        cs = t.get("consensus_score")
        if cs is not None and cs < min_consensus:
            continue
        if _would_take_trade(t, params):
            accepted.append(t["pnl_pct"])

    return np.array(accepted, dtype=np.float64) if accepted else np.array([], dtype=np.float64)


# ---------------------------------------------------------------------------
# Optuna mutation (challenger param discovery on in-sample data)
# ---------------------------------------------------------------------------

# Import the search-space builder from the canonical recursive engine so
# the challenger explores the exact same parameter topology.
from src.core.recursive_engine import _suggest_params


def _run_challenger_optuna(
    in_sample_trades: list[dict[str, Any]],
    n_trials: int = _CHALLENGER_N_TRIALS,
) -> dict[str, Any]:
    """Run a short Optuna study on *in_sample_trades* to discover challenger params."""
    storage = optuna.storages.RDBStorage(
        url=SYNC_POSTGRES_URL,
        engine_kwargs={"pool_pre_ping": True, "pool_size": 3},
    )

    study = optuna.create_study(
        study_name=f"{_CHALLENGER_STUDY_NAME}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        returns = _filter_returns(in_sample_trades, params)
        if len(returns) < _MIN_ACCEPTED_TRADES:
            return _INSUFFICIENT_TRADES_PENALTY
        return _compute_sortino(returns)

    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    logger.info(
        "Challenger Optuna complete — best Sortino=%.4f (trial #%d, %d trials)",
        best.value,
        best.number,
        n_trials,
    )
    return dict(best.params)


def _load_champion_params() -> dict[str, Any]:
    """Load the current champion params from the production Optuna study."""
    storage = optuna.storages.RDBStorage(
        url=SYNC_POSTGRES_URL,
        engine_kwargs={"pool_pre_ping": True, "pool_size": 3},
    )

    try:
        study = optuna.load_study(
            study_name=_CHAMPION_STUDY_NAME,
            storage=storage,
        )
        return dict(study.best_params)
    except KeyError:
        logger.warning(
            "No existing champion study '%s' found — using empty params",
            _CHAMPION_STUDY_NAME,
        )
        return {}


# ---------------------------------------------------------------------------
# Celery tasks — the two parallel legs of the chord
# ---------------------------------------------------------------------------


@celery_app.task(
    name="src.core.champion_challenger.run_champion_sim",
    bind=True,
    max_retries=1,
    default_retry_delay=30,
)
def run_champion_sim(self) -> dict[str, Any]:
    """Evaluate the current champion parameters on the OOS holdout.

    Returns a serialisable dict containing the champion's parameter set,
    its block-level Sortino samples, and the aggregate Sortino ratio.
    """
    try:
        all_trades = _load_recent_trades(_TOTAL_LOOKBACK)
        if len(all_trades) < _OOS_SIZE + _MIN_ACCEPTED_TRADES:
            return {
                "role": "champion",
                "status": "insufficient_data",
                "total_trades": len(all_trades),
            }

        oos_trades = all_trades[-_OOS_SIZE:]
        champion_params = _load_champion_params()

        if not champion_params:
            return {
                "role": "champion",
                "status": "no_champion_study",
                "params": {},
            }

        oos_returns = _filter_returns(oos_trades, champion_params)
        aggregate_sortino = _compute_sortino(oos_returns)
        block_sortinos = _compute_block_sortinos(oos_returns, _BLOCK_SIZE)

        logger.info(
            "Champion OOS: aggregate_sortino=%.4f, %d blocks, %d accepted/%d total",
            aggregate_sortino,
            len(block_sortinos),
            len(oos_returns),
            len(oos_trades),
        )

        return {
            "role": "champion",
            "status": "ok",
            "params": champion_params,
            "aggregate_sortino": round(aggregate_sortino, 6),
            "block_sortinos": block_sortinos.tolist(),
            "accepted_trades": len(oos_returns),
            "oos_size": len(oos_trades),
        }

    except Exception as exc:
        logger.exception("Champion simulation failed")
        raise self.retry(exc=exc)


@celery_app.task(
    name="src.core.champion_challenger.run_challenger_sim",
    bind=True,
    max_retries=1,
    default_retry_delay=30,
)
def run_challenger_sim(self) -> dict[str, Any]:
    """Discover challenger params via Optuna on IS data, evaluate on OOS.

    The Optuna mutation runs exclusively on the in-sample partition
    (first 800 trades) so the out-of-sample evaluation (last 200) is
    completely uncontaminated.
    """
    try:
        all_trades = _load_recent_trades(_TOTAL_LOOKBACK)
        if len(all_trades) < _OOS_SIZE + _MIN_ACCEPTED_TRADES:
            return {
                "role": "challenger",
                "status": "insufficient_data",
                "total_trades": len(all_trades),
            }

        is_trades = all_trades[: -_OOS_SIZE]
        oos_trades = all_trades[-_OOS_SIZE:]

        # Phase 1: discover challenger parameters on in-sample data.
        challenger_params = _run_challenger_optuna(is_trades, _CHALLENGER_N_TRIALS)

        # Phase 2: evaluate on untouched OOS holdout.
        oos_returns = _filter_returns(oos_trades, challenger_params)
        aggregate_sortino = _compute_sortino(oos_returns)
        block_sortinos = _compute_block_sortinos(oos_returns, _BLOCK_SIZE)

        logger.info(
            "Challenger OOS: aggregate_sortino=%.4f, %d blocks, %d accepted/%d total",
            aggregate_sortino,
            len(block_sortinos),
            len(oos_returns),
            len(oos_trades),
        )

        return {
            "role": "challenger",
            "status": "ok",
            "params": challenger_params,
            "aggregate_sortino": round(aggregate_sortino, 6),
            "block_sortinos": block_sortinos.tolist(),
            "accepted_trades": len(oos_returns),
            "oos_size": len(oos_trades),
        }

    except Exception as exc:
        logger.exception("Challenger simulation failed")
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Chord callback — statistical comparison and conditional promotion
# ---------------------------------------------------------------------------


def _promote_to_champion(params: dict[str, Any]) -> dict[str, Any]:
    """Persist the promoted challenger as the new champion study best trial.

    Uses ``study.enqueue_trial`` + ``study.optimize(n_trials=1)`` to
    inject the challenger's params as a completed trial into the
    champion study, making it the new ``best_trial``.

    For live Redis hot-swap, the downstream nightly optimizer will pick
    up the updated best_params on its next run, or an operator can
    trigger ``RedisStateManager.hot_swap_strategy_params()`` manually.
    """
    storage = optuna.storages.RDBStorage(
        url=SYNC_POSTGRES_URL,
        engine_kwargs={"pool_pre_ping": True, "pool_size": 3},
    )

    study = optuna.create_study(
        study_name=_CHAMPION_STUDY_NAME,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    # Enqueue the exact challenger params so the next trial evaluates them.
    study.enqueue_trial(params)

    # Load trades and evaluate the enqueued trial so it becomes a
    # completed trial with a real objective value.
    trades = _load_recent_trades(_TOTAL_LOOKBACK)

    def objective(trial: optuna.Trial) -> float:
        p = _suggest_params(trial)
        returns = _filter_returns(trades, p)
        if len(returns) < _MIN_ACCEPTED_TRADES:
            return _INSUFFICIENT_TRADES_PENALTY
        return _compute_sortino(returns)

    study.optimize(objective, n_trials=1)

    new_best = study.best_trial
    logger.info(
        "Champion promoted — new best Sortino=%.4f (trial #%d)",
        new_best.value,
        new_best.number,
    )

    return {
        "promoted_sortino": round(new_best.value, 6),
        "promoted_trial": new_best.number,
        "promoted_params": dict(new_best.params),
    }


@celery_app.task(
    name="src.core.champion_challenger.evaluate_promotion",
    bind=True,
    max_retries=0,
)
def evaluate_promotion(self, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Chord callback: perform Welch's T-test and conditionally promote.

    Receives a list of two dicts — the champion and challenger simulation
    results produced by ``run_champion_sim`` and ``run_challenger_sim``.

    The Independent-Samples Welch's T-test compares the block-level
    Sortino ratio distributions.  Promotion occurs **only** when:

    1. The Challenger's mean block Sortino **exceeds** the Champion's.
    2. The improvement is at least ``_MIN_SORTINO_IMPROVEMENT``.
    3. The difference is statistically significant at *p < 0.05*
       (one-tailed: Challenger > Champion).

    Returns
    -------
    dict
        Full audit trail: means, t-statistic, p-value, promotion decision,
        and (if promoted) the new champion parameters.
    """
    # --- Parse the two results ------------------------------------------------
    champion_result: dict[str, Any] | None = None
    challenger_result: dict[str, Any] | None = None

    for r in results:
        if r.get("role") == "champion":
            champion_result = r
        elif r.get("role") == "challenger":
            challenger_result = r

    if champion_result is None or challenger_result is None:
        logger.error(
            "Chord callback received incomplete results: %s",
            [r.get("role") for r in results],
        )
        return {"decision": "ERROR", "reason": "incomplete_results"}

    # --- Guard: both legs must have succeeded ---------------------------------
    if champion_result.get("status") != "ok":
        logger.warning(
            "Champion leg failed (status=%s) — aborting evaluation",
            champion_result.get("status"),
        )
        return {
            "decision": "ABORT",
            "reason": f"champion_{champion_result.get('status')}",
        }

    if challenger_result.get("status") != "ok":
        logger.warning(
            "Challenger leg failed (status=%s) — aborting evaluation",
            challenger_result.get("status"),
        )
        return {
            "decision": "ABORT",
            "reason": f"challenger_{challenger_result.get('status')}",
        }

    champion_blocks = np.array(champion_result["block_sortinos"], dtype=np.float64)
    challenger_blocks = np.array(challenger_result["block_sortinos"], dtype=np.float64)

    champion_mean = float(np.mean(champion_blocks))
    challenger_mean = float(np.mean(challenger_blocks))
    improvement = challenger_mean - champion_mean

    audit: dict[str, Any] = {
        "champion_aggregate_sortino": champion_result["aggregate_sortino"],
        "challenger_aggregate_sortino": challenger_result["aggregate_sortino"],
        "champion_block_mean": round(champion_mean, 6),
        "challenger_block_mean": round(challenger_mean, 6),
        "sortino_improvement": round(improvement, 6),
        "champion_accepted_trades": champion_result["accepted_trades"],
        "challenger_accepted_trades": challenger_result["accepted_trades"],
        "champion_blocks": len(champion_blocks),
        "challenger_blocks": len(challenger_blocks),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    # --- Gate 1: Challenger must actually be better ---------------------------
    if improvement <= 0:
        audit["decision"] = "RETAIN_CHAMPION"
        audit["reason"] = "challenger_not_better"
        logger.info(
            "Champion retained — Challenger Sortino (%.4f) <= Champion (%.4f)",
            challenger_mean,
            champion_mean,
        )
        return audit

    # --- Gate 2: Minimum practical improvement --------------------------------
    if improvement < _MIN_SORTINO_IMPROVEMENT:
        audit["decision"] = "RETAIN_CHAMPION"
        audit["reason"] = (
            f"improvement_below_threshold "
            f"({improvement:.4f} < {_MIN_SORTINO_IMPROVEMENT})"
        )
        logger.info(
            "Champion retained — improvement %.4f below threshold %.4f",
            improvement,
            _MIN_SORTINO_IMPROVEMENT,
        )
        return audit

    # --- Gate 3: Statistical significance via Welch's T-test ------------------
    if len(champion_blocks) < 2 or len(challenger_blocks) < 2:
        audit["decision"] = "RETAIN_CHAMPION"
        audit["reason"] = "insufficient_blocks_for_ttest"
        audit["t_statistic"] = None
        audit["p_value"] = None
        logger.warning(
            "Champion retained — not enough blocks for T-test "
            "(champion=%d, challenger=%d)",
            len(champion_blocks),
            len(challenger_blocks),
        )
        return audit

    # Welch's T-test (unequal variances, independent samples).
    t_stat, two_tailed_p = stats.ttest_ind(
        challenger_blocks,
        champion_blocks,
        equal_var=False,
        alternative="greater",
    )

    audit["t_statistic"] = round(float(t_stat), 6)
    audit["p_value"] = round(float(two_tailed_p), 8)

    if two_tailed_p >= _P_VALUE_THRESHOLD:
        audit["decision"] = "RETAIN_CHAMPION"
        audit["reason"] = (
            f"not_significant (p={two_tailed_p:.6f} >= {_P_VALUE_THRESHOLD})"
        )
        logger.info(
            "Champion retained — improvement not significant "
            "(t=%.4f, p=%.6f >= %.2f)",
            t_stat,
            two_tailed_p,
            _P_VALUE_THRESHOLD,
        )
        return audit

    # --- All gates passed: PROMOTE --------------------------------------------
    logger.info(
        "PROMOTION: Challenger wins — improvement=%.4f, t=%.4f, p=%.6f",
        improvement,
        t_stat,
        two_tailed_p,
    )

    try:
        promotion_result = _promote_to_champion(challenger_result["params"])
        audit["decision"] = "PROMOTE_CHALLENGER"
        audit["promotion"] = promotion_result
    except Exception:
        logger.exception("Promotion failed — Champion retained despite passing T-test")
        audit["decision"] = "PROMOTION_FAILED"
        audit["reason"] = "exception_during_promotion"

    return audit


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------


@celery_app.task(
    name="src.core.champion_challenger.trigger_champion_challenger",
)
def trigger_champion_challenger() -> dict[str, str]:
    """Fire the Champion–Challenger chord.

    Launches ``run_champion_sim`` and ``run_challenger_sim`` in parallel
    via a Celery ``chord``.  When both complete, the
    ``evaluate_promotion`` callback runs automatically to perform the
    statistical test and conditional promotion.

    Returns
    -------
    dict
        Acknowledgement with the chord task ID for tracking.
    """
    logger.info("Triggering Champion–Challenger chord")

    callback = evaluate_promotion.s()
    result = chord(
        [run_champion_sim.s(), run_challenger_sim.s()],
    )(callback)

    logger.info("Chord dispatched — callback task ID: %s", result.id)
    return {
        "status": "chord_dispatched",
        "callback_task_id": result.id,
        "dispatched_at": datetime.now(timezone.utc).isoformat(),
    }
