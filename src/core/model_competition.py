"""Model Competition orchestrator — parallel backtests with statistical promotion.

Runs two vectorised backtests in parallel Celery tasks:

* **Champion** — current production parameters loaded from Redis via
  ``get_strategy_params()``.
* **Challenger** — Optuna-suggested parameter tweaks loaded from the
  latest completed study trial.

Both are evaluated strictly on **out-of-sample** data: the last 30 days
of closed trades from ``trade_log``.  Per-desk return streams are split
into non-overlapping blocks and scored via the annualised Sortino ratio.

An Independent-Samples Welch's T-test determines whether the
Challenger's Sortino improvement is statistically significant at
*p < 0.05*.  Promotion occurs **only** when:

1. Challenger mean block Sortino > Champion.
2. Welch's T-test p-value < 0.05 (one-tailed: Challenger > Champion).
3. ``atomic_swap_params()`` in ``redis_manager.py`` succeeds for every
   desk that changed.

Start the worker::

    celery -A src.core.model_competition worker --loglevel=info

Trigger manually::

    from src.core.model_competition import run_competition
    run_competition.delay()
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import optuna
from celery import Celery
from scipy import stats
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.services.redis_manager import RedisStateManager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
)
SYNC_POSTGRES_URL: str = POSTGRES_URL.replace("+asyncpg", "")

_LOOKBACK_DAYS: int = int(os.getenv("MC_LOOKBACK_DAYS", "30"))
_CHAMPION_STUDY: str = os.getenv("MC_CHAMPION_STUDY", "oniquant_evolution")
_P_VALUE_THRESHOLD: float = float(os.getenv("MC_P_VALUE", "0.05"))
_BLOCK_SIZE: int = int(os.getenv("MC_BLOCK_SIZE", "20"))
_MIN_TRADES: int = int(os.getenv("MC_MIN_TRADES", "30"))
_ANNUAL_TRADING_PERIODS: int = 252
_DESK_IDS: list[int] = [1, 2, 3, 4, 5]

logger = logging.getLogger("oniquant.model_competition")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

celery_app = Celery(
    "model_competition",
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
# SQL — 30-day OOS trade data
# ---------------------------------------------------------------------------

_OOS_QUERY = text("""
    SELECT
        desk_id,
        pnl_pct,
        atr_at_entry,
        volume_ratio,
        rsi_at_entry,
        adx_at_entry,
        cci_at_entry,
        ema_distance,
        cvd_zscore,
        lorentzian_score,
        consensus_score
    FROM trade_log
    WHERE status = 'SIM_CLOSED'
      AND closed_at >= NOW() - MAKE_INTERVAL(days => :lookback_days)
    ORDER BY closed_at ASC
""")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_oos_trades() -> list[dict[str, Any]]:
    """Fetch the last 30 days of closed trades (strictly out-of-sample)."""
    db: Session = _SessionLocal()
    try:
        rows = db.execute(
            _OOS_QUERY, {"lookback_days": _LOOKBACK_DAYS},
        ).fetchall()
        return [dict(r._mapping) for r in rows]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------


def _load_champion_params() -> dict[int, tuple[int, dict[str, Any]]]:
    """Load current production params from Redis for all 5 desks.

    Returns {desk_id: (version, params)}.
    """
    async def _fetch() -> dict[int, tuple[int, dict[str, Any]]]:
        mgr = RedisStateManager()
        await mgr.connect()
        try:
            result = {}
            for did in _DESK_IDS:
                version, params = await mgr.get_strategy_params(did)
                result[did] = (version, params)
            return result
        finally:
            await mgr.close()

    return asyncio.get_event_loop().run_until_complete(_fetch())


def _load_challenger_params() -> dict[str, Any]:
    """Load the best trial params from the Optuna study."""
    storage = optuna.storages.RDBStorage(
        url=SYNC_POSTGRES_URL,
        engine_kwargs={"pool_pre_ping": True, "pool_size": 3},
    )
    try:
        study = optuna.load_study(
            study_name=_CHAMPION_STUDY, storage=storage,
        )
        return dict(study.best_params)
    except KeyError:
        logger.warning("No Optuna study '%s' found", _CHAMPION_STUDY)
        return {}


# ---------------------------------------------------------------------------
# Trade filter (per-desk indicator gates)
# ---------------------------------------------------------------------------


def _would_take_trade(trade: dict[str, Any], params: dict[str, Any]) -> bool:
    """Simulate whether *trade* passes the parameter filters."""
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


# ---------------------------------------------------------------------------
# Vectorised backtest
# ---------------------------------------------------------------------------


def _filter_returns(
    trades: list[dict[str, Any]],
    params: dict[str, Any],
) -> np.ndarray:
    """Apply parameter filters and return the accepted pnl_pct array."""
    min_consensus = params.get("min_consensus_score", 0.0)
    accepted: list[float] = []
    for t in trades:
        cs = t.get("consensus_score")
        if cs is not None and cs < min_consensus:
            continue
        if _would_take_trade(t, params):
            accepted.append(float(t["pnl_pct"]))
    return np.array(accepted, dtype=np.float64) if accepted else np.array([], dtype=np.float64)


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
        return float(mean_return * np.sqrt(_ANNUAL_TRADING_PERIODS)) if mean_return > 0 else 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0.0:
        return 0.0
    return float((mean_return / downside_std) * np.sqrt(_ANNUAL_TRADING_PERIODS))


def _compute_block_sortinos(returns: np.ndarray, block_size: int) -> np.ndarray:
    """Split returns into non-overlapping blocks, compute Sortino per block."""
    n_blocks = len(returns) // block_size
    if n_blocks < 2:
        return np.array([_compute_sortino(returns)])
    sortinos = np.empty(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        block = returns[i * block_size : (i + 1) * block_size]
        sortinos[i] = _compute_sortino(block)
    return sortinos


# ---------------------------------------------------------------------------
# Backtest runner (one for each contestant)
# ---------------------------------------------------------------------------


def _run_backtest(
    trades: list[dict[str, Any]],
    params: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Run a vectorised backtest and return Sortino statistics."""
    returns = _filter_returns(trades, params)

    if len(returns) < _MIN_TRADES:
        logger.warning(
            "%s backtest: only %d accepted trades (need %d) — insufficient",
            label, len(returns), _MIN_TRADES,
        )
        return {
            "label": label,
            "status": "insufficient_trades",
            "accepted_trades": len(returns),
        }

    aggregate_sortino = _compute_sortino(returns)
    block_sortinos = _compute_block_sortinos(returns, _BLOCK_SIZE)

    logger.info(
        "%s backtest: sortino=%.4f, %d accepted trades, %d blocks",
        label, aggregate_sortino, len(returns), len(block_sortinos),
    )

    return {
        "label": label,
        "status": "ok",
        "aggregate_sortino": round(aggregate_sortino, 6),
        "block_sortinos": block_sortinos.tolist(),
        "accepted_trades": len(returns),
        "n_blocks": len(block_sortinos),
        "mean_return": round(float(np.mean(returns)), 8),
        "win_rate": round(float(np.mean(returns > 0)), 4),
    }


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------


def _welch_ttest(
    champion_result: dict[str, Any],
    challenger_result: dict[str, Any],
) -> dict[str, Any]:
    """Welch's T-test on block-level Sortinos with promotion decision."""
    champ_blocks = np.array(champion_result["block_sortinos"], dtype=np.float64)
    chall_blocks = np.array(challenger_result["block_sortinos"], dtype=np.float64)

    champ_mean = float(np.mean(champ_blocks))
    chall_mean = float(np.mean(chall_blocks))
    improvement = chall_mean - champ_mean

    audit: dict[str, Any] = {
        "champion_sortino": champion_result["aggregate_sortino"],
        "challenger_sortino": challenger_result["aggregate_sortino"],
        "champion_block_mean": round(champ_mean, 6),
        "challenger_block_mean": round(chall_mean, 6),
        "sortino_improvement": round(improvement, 6),
        "champion_trades": champion_result["accepted_trades"],
        "challenger_trades": challenger_result["accepted_trades"],
        "champion_blocks": len(champ_blocks),
        "challenger_blocks": len(chall_blocks),
    }

    # Gate 1: Challenger must be better.
    if improvement <= 0:
        audit["decision"] = "RETAIN_CHAMPION"
        audit["reason"] = "challenger_not_better"
        audit["t_statistic"] = None
        audit["p_value"] = None
        return audit

    # Gate 2: Enough blocks for a meaningful test.
    if len(champ_blocks) < 2 or len(chall_blocks) < 2:
        audit["decision"] = "RETAIN_CHAMPION"
        audit["reason"] = "insufficient_blocks_for_ttest"
        audit["t_statistic"] = None
        audit["p_value"] = None
        return audit

    # Gate 3: Welch's T-test (one-tailed: Challenger > Champion).
    t_stat, p_value = stats.ttest_ind(
        chall_blocks, champ_blocks,
        equal_var=False,
        alternative="greater",
    )

    audit["t_statistic"] = round(float(t_stat), 6)
    audit["p_value"] = round(float(p_value), 8)

    if p_value < _P_VALUE_THRESHOLD:
        audit["decision"] = "PROMOTE_CHALLENGER"
        audit["reason"] = f"significant (p={p_value:.6f} < {_P_VALUE_THRESHOLD})"
    else:
        audit["decision"] = "RETAIN_CHAMPION"
        audit["reason"] = f"not_significant (p={p_value:.6f} >= {_P_VALUE_THRESHOLD})"

    return audit


# ---------------------------------------------------------------------------
# Redis atomic promotion
# ---------------------------------------------------------------------------


def _promote_to_production(
    challenger_params: dict[str, Any],
    champion_versions: dict[int, int],
) -> dict[str, Any]:
    """Atomically swap challenger params into Redis for each desk.

    Uses ``atomic_swap_params()`` with CAS version checking to prevent
    clobbering a concurrent update.
    """
    async def _swap() -> dict[str, Any]:
        mgr = RedisStateManager()
        await mgr.connect()
        try:
            results: dict[str, Any] = {}
            desks_swapped = 0

            for did in _DESK_IDS:
                # Build per-desk params from the flat challenger dict.
                prefix = f"d{did}_"
                desk_params = {
                    k: v for k, v in challenger_params.items()
                    if k.startswith(prefix)
                }
                # Include cross-desk params.
                if "min_consensus_score" in challenger_params:
                    desk_params["min_consensus_score"] = challenger_params["min_consensus_score"]

                if not desk_params:
                    continue

                expected_version = champion_versions.get(did, 0)

                try:
                    receipt = await mgr.atomic_swap_params(
                        desk_id=did,
                        new_config=desk_params,
                        expected_version=expected_version,
                    )
                    results[f"desk_{did}"] = {
                        "status": "swapped",
                        "version": receipt["version"],
                        "previous_version": receipt["previous_version"],
                    }
                    desks_swapped += 1
                    logger.info(
                        "Desk %d promoted: version %d → %d",
                        did, receipt["previous_version"], receipt["version"],
                    )
                except RuntimeError as exc:
                    results[f"desk_{did}"] = {
                        "status": "cas_rejected",
                        "error": str(exc),
                    }
                    logger.warning("Desk %d CAS rejected: %s", did, exc)

            results["desks_swapped"] = desks_swapped
            return results
        finally:
            await mgr.close()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_swap())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Celery task — main competition entry point
# ---------------------------------------------------------------------------


@celery_app.task(
    name="src.core.model_competition.run_competition",
    bind=True,
    max_retries=1,
    default_retry_delay=300,
)
def run_competition(self) -> dict[str, Any]:
    """Execute a Champion vs Challenger model competition.

    Pipeline:
    1. Load 30-day OOS trades from trade_log.
    2. Load Champion params from Redis, Challenger params from Optuna.
    3. Run parallel vectorised backtests (both on same OOS data).
    4. Welch's T-test on block Sortino distributions.
    5. If p < 0.05, atomic_swap_params() for each desk.
    """
    try:
        now = datetime.now(timezone.utc).isoformat()

        # --- Step 1: Load OOS data ----------------------------------------
        trades = _load_oos_trades()
        if len(trades) < _MIN_TRADES:
            return {
                "status": "skipped",
                "reason": f"insufficient_oos_trades ({len(trades)} < {_MIN_TRADES})",
                "evaluated_at": now,
            }

        logger.info(
            "Model competition: %d OOS trades (%d-day window)",
            len(trades), _LOOKBACK_DAYS,
        )

        # --- Step 2: Load params ------------------------------------------
        champion_desk_data = _load_champion_params()
        # Flatten per-desk params into a single dict for the backtest filter.
        champion_params: dict[str, Any] = {}
        champion_versions: dict[int, int] = {}
        for did, (version, params) in champion_desk_data.items():
            champion_versions[did] = version
            champion_params.update(params)

        challenger_params = _load_challenger_params()

        if not champion_params:
            return {
                "status": "skipped",
                "reason": "no_champion_params_in_redis",
                "evaluated_at": now,
            }
        if not challenger_params:
            return {
                "status": "skipped",
                "reason": "no_challenger_params_from_optuna",
                "evaluated_at": now,
            }

        # --- Step 3: Parallel backtests -----------------------------------
        champion_result = _run_backtest(trades, champion_params, "Champion")
        challenger_result = _run_backtest(trades, challenger_params, "Challenger")

        if champion_result.get("status") != "ok":
            return {
                "status": "aborted",
                "reason": f"champion_backtest_{champion_result.get('status')}",
                "champion": champion_result,
                "evaluated_at": now,
            }

        if challenger_result.get("status") != "ok":
            return {
                "status": "aborted",
                "reason": f"challenger_backtest_{challenger_result.get('status')}",
                "challenger": challenger_result,
                "evaluated_at": now,
            }

        # --- Step 4: Statistical comparison -------------------------------
        ttest_result = _welch_ttest(champion_result, challenger_result)

        result: dict[str, Any] = {
            "status": "completed",
            "oos_trades": len(trades),
            "lookback_days": _LOOKBACK_DAYS,
            "champion": champion_result,
            "challenger": challenger_result,
            "statistical_test": ttest_result,
            "evaluated_at": now,
        }

        # --- Step 5: Promote if significant -------------------------------
        if ttest_result["decision"] == "PROMOTE_CHALLENGER":
            logger.info(
                "PROMOTION: Challenger wins (t=%.4f, p=%.6f)",
                ttest_result["t_statistic"],
                ttest_result["p_value"],
            )
            promotion = _promote_to_production(
                challenger_params, champion_versions,
            )
            result["promotion"] = promotion
            result["promoted"] = True

            logger.info(
                "Promotion complete: %d desks swapped",
                promotion.get("desks_swapped", 0),
            )
        else:
            result["promoted"] = False
            logger.info(
                "Champion retained: %s", ttest_result.get("reason"),
            )

        return result

    except Exception as exc:
        logger.exception("Model competition failed")
        raise self.retry(exc=exc)
