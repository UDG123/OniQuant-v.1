"""Walk-Forward Analysis for Champion vs. Challenger parameter validation.

Partitions ``ml_training_data`` into three sequential time-ordered
blocks — Train, Validate, Out-of-Sample — runs a vectorized simulation
for both parameter sets across all blocks, and applies a Welch t-test
on the OOS block to determine whether the Challenger's Sortino
improvement is statistically significant (p < 0.05).

Usage::

    from src.core.validation import walk_forward_validate

    report = walk_forward_validate(
        trades=trades,                 # list[dict] from ml_training_data
        challenger_params=new_params,  # from Optuna best_trial
        champion_params=current_params,
    )
    if report.promote_challenger:
        # hot-swap into Redis
        ...
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

logger = logging.getLogger("oniquant.core.validation")

_ANNUAL_TRADING_PERIODS: int = 252
_MIN_OOS_TRADES: int = 15
_SIGNIFICANCE_LEVEL: float = 0.05


# ---------------------------------------------------------------------------
# Enums & Pydantic models
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    """Outcome of the walk-forward validation."""

    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class BlockMetrics(BaseModel):
    """Performance metrics for a single data block (Train / Validate / OOS)."""

    block_name: str = Field(..., description="Train, Validate, or OOS")
    n_trades_total: int = Field(..., description="Trades in the block")
    n_trades_accepted: int = Field(..., description="Trades passing param filter")
    acceptance_rate: float = Field(..., description="accepted / total")
    sortino_ratio: float = Field(..., description="Annualised Sortino (MAR=0)")
    mean_return: float = Field(..., description="Arithmetic mean of accepted returns")
    win_rate: float = Field(..., description="Fraction of accepted trades with pnl > 0")
    max_drawdown_pct: float = Field(
        ..., description="Maximum peak-to-trough drawdown on the equity curve",
    )


class ParamSetReport(BaseModel):
    """Full walk-forward results for a single parameter set."""

    label: str = Field(..., description="'champion' or 'challenger'")
    params: dict[str, Any]
    train: BlockMetrics
    validate_block: BlockMetrics
    oos: BlockMetrics


class ValidationReport(BaseModel):
    """Complete Walk-Forward Analysis report comparing Champion vs. Challenger."""

    verdict: Verdict
    promote_challenger: bool = Field(
        ..., description="True when Challenger passes all gates",
    )
    t_statistic: float = Field(
        ..., description="Welch t-test statistic (Challenger - Champion OOS returns)",
    )
    p_value: float = Field(
        ..., description="Two-sided p-value from Welch t-test",
    )
    significance_level: float = Field(
        default=_SIGNIFICANCE_LEVEL,
        description="Required p-value threshold for promotion",
    )
    challenger: ParamSetReport
    champion: ParamSetReport
    train_pct: float = Field(..., description="Fraction of data used for training")
    validate_pct: float = Field(..., description="Fraction of data used for validation")
    oos_pct: float = Field(..., description="Fraction of data used for OOS")
    total_trades: int
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Sortino ratio (vectorised)
# ---------------------------------------------------------------------------


def _compute_sortino(returns: np.ndarray) -> float:
    """Annualised Sortino ratio (MAR = 0).

    Matches the implementation in ``src.core.optimizer`` exactly.
    """
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


# ---------------------------------------------------------------------------
# Max drawdown (vectorised via cumulative max)
# ---------------------------------------------------------------------------


def _max_drawdown(returns: np.ndarray) -> float:
    """Peak-to-trough drawdown on a cumulative equity curve.

    Returns 0.0 when the curve is monotonically increasing or has
    fewer than 2 points.
    """
    if len(returns) < 2:
        return 0.0

    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)

    drawdowns = (running_max - equity) / running_max
    return float(np.max(drawdowns))


# ---------------------------------------------------------------------------
# Trade filter — mirrors src.core.optimizer._would_take_trade
# ---------------------------------------------------------------------------


def _would_take_trade(trade: dict[str, Any], params: dict[str, Any]) -> bool:
    """Simulate whether *trade* passes the candidate parameter filters.

    Identical logic to ``src.core.optimizer._would_take_trade`` so that
    walk-forward results are directly comparable to Optuna trials.
    """
    desk = trade["desk_id"]

    if desk == 1:
        cvd = trade.get("cvd_zscore")
        if cvd is not None and abs(cvd) < params.get("d1_ofi_threshold", 0):
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None:
            lo = params.get("d1_rsi_lower", 0)
            hi = params.get("d1_rsi_upper", 100)
            if not (lo < rsi < hi):
                return False
        return True

    if desk == 2:
        cvd = trade.get("cvd_zscore")
        if cvd is not None and abs(cvd) < params.get("d2_kalman_z_entry", 0):
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None:
            lo = params.get("d2_rsi_oversold", 0)
            hi = params.get("d2_rsi_overbought", 100)
            if not (lo < rsi < hi):
                return False
        return True

    if desk == 3:
        score = trade.get("lorentzian_score")
        if score is not None and score < params.get("d3_lorentzian_threshold", 0):
            return False
        rsi = trade.get("rsi_at_entry")
        if rsi is not None:
            lo = params.get("d3_rsi_lower", 0)
            hi = params.get("d3_rsi_upper", 100)
            if not (lo < rsi < hi):
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
        if rsi is not None:
            lo = params.get("d5_rsi_lower", 0)
            hi = params.get("d5_rsi_upper", 100)
            if not (lo < rsi < hi):
                return False
        return True

    return True


# ---------------------------------------------------------------------------
# Vectorised block simulation
# ---------------------------------------------------------------------------


def _simulate_block(
    trades: list[dict[str, Any]],
    params: dict[str, Any],
    block_name: str,
) -> tuple[BlockMetrics, np.ndarray]:
    """Run a vectorised simulation over a trade block.

    Applies the consensus-score gate (``min_consensus_score``) and
    per-desk indicator filters, then computes Sortino, win rate,
    max drawdown, and mean return over the accepted subset.

    Returns
    -------
    tuple[BlockMetrics, np.ndarray]
        Computed metrics and the raw accepted-returns array (needed
        downstream for the t-test).
    """
    min_consensus = params.get("min_consensus_score", 0.0)
    n_total = len(trades)

    accepted_returns: list[float] = []
    for t in trades:
        cs = t.get("consensus_score")
        if cs is not None and cs < min_consensus:
            continue
        if _would_take_trade(t, params):
            accepted_returns.append(t["pnl_pct"])

    returns = np.array(accepted_returns, dtype=np.float64) if accepted_returns else np.array([], dtype=np.float64)
    n_accepted = len(returns)

    sortino = _compute_sortino(returns)
    mean_ret = float(np.mean(returns)) if n_accepted > 0 else 0.0
    win = float(np.mean(returns > 0)) if n_accepted > 0 else 0.0
    mdd = _max_drawdown(returns) if n_accepted >= 2 else 0.0
    acc_rate = n_accepted / n_total if n_total > 0 else 0.0

    metrics = BlockMetrics(
        block_name=block_name,
        n_trades_total=n_total,
        n_trades_accepted=n_accepted,
        acceptance_rate=round(acc_rate, 6),
        sortino_ratio=round(sortino, 6),
        mean_return=round(mean_ret, 8),
        win_rate=round(win, 6),
        max_drawdown_pct=round(mdd, 6),
    )

    return metrics, returns


# ---------------------------------------------------------------------------
# Data partitioning
# ---------------------------------------------------------------------------


def _partition_trades(
    trades: list[dict[str, Any]],
    train_pct: float,
    validate_pct: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split trades into sequential Train / Validate / OOS blocks.

    Trades are assumed to be pre-sorted by ``closed_at`` ascending
    (oldest first).  The split is positional — no shuffling — to
    preserve temporal ordering for walk-forward validity.
    """
    n = len(trades)
    train_end = int(n * train_pct)
    validate_end = train_end + int(n * validate_pct)

    return trades[:train_end], trades[train_end:validate_end], trades[validate_end:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def walk_forward_validate(
    trades: list[dict[str, Any]],
    challenger_params: dict[str, Any],
    champion_params: dict[str, Any],
    *,
    train_pct: float = 0.50,
    validate_pct: float = 0.25,
    significance_level: float = _SIGNIFICANCE_LEVEL,
) -> ValidationReport:
    """Run a Walk-Forward Analysis comparing Challenger vs. Champion.

    Parameters
    ----------
    trades : list[dict]
        Rows from ``ml_training_data`` sorted by ``closed_at`` ascending.
        Each dict must contain at least ``desk_id``, ``pnl_pct``, and
        the indicator columns used by the trade filter.
    challenger_params : dict
        Candidate parameter set produced by Optuna.
    champion_params : dict
        Current production parameter set.
    train_pct : float
        Fraction of trades allocated to the training block (default 0.50).
    validate_pct : float
        Fraction allocated to validation (default 0.25).
        The remainder (``1 - train_pct - validate_pct``) becomes OOS.
    significance_level : float
        p-value threshold for declaring statistical significance
        (default 0.05).

    Returns
    -------
    ValidationReport
        Pydantic model with per-block metrics for both param sets,
        t-test results, and a promotion verdict.
    """
    oos_pct = round(1.0 - train_pct - validate_pct, 6)

    # Sort by closed_at ascending to guarantee temporal ordering.
    sorted_trades = sorted(
        trades,
        key=lambda t: t.get("closed_at", ""),
    )

    train_block, validate_block, oos_block = _partition_trades(
        sorted_trades, train_pct, validate_pct,
    )

    logger.info(
        "Walk-forward partition: train=%d validate=%d oos=%d (total=%d)",
        len(train_block),
        len(validate_block),
        len(oos_block),
        len(sorted_trades),
    )

    # --- Simulate both param sets across all three blocks ---
    ch_train, _ = _simulate_block(train_block, champion_params, "Train")
    ch_val, _ = _simulate_block(validate_block, champion_params, "Validate")
    ch_oos, ch_oos_returns = _simulate_block(oos_block, champion_params, "OOS")

    cl_train, _ = _simulate_block(train_block, challenger_params, "Train")
    cl_val, _ = _simulate_block(validate_block, challenger_params, "Validate")
    cl_oos, cl_oos_returns = _simulate_block(oos_block, challenger_params, "OOS")

    # --- Statistical test on OOS returns ---
    # Check if we have enough OOS trades for a meaningful test.
    if len(ch_oos_returns) < _MIN_OOS_TRADES or len(cl_oos_returns) < _MIN_OOS_TRADES:
        logger.warning(
            "Insufficient OOS trades for t-test: champion=%d challenger=%d (min=%d)",
            len(ch_oos_returns),
            len(cl_oos_returns),
            _MIN_OOS_TRADES,
        )
        return ValidationReport(
            verdict=Verdict.INSUFFICIENT_DATA,
            promote_challenger=False,
            t_statistic=0.0,
            p_value=1.0,
            significance_level=significance_level,
            challenger=ParamSetReport(
                label="challenger",
                params=challenger_params,
                train=cl_train,
                validate_block=cl_val,
                oos=cl_oos,
            ),
            champion=ParamSetReport(
                label="champion",
                params=champion_params,
                train=ch_train,
                validate_block=ch_val,
                oos=ch_oos,
            ),
            train_pct=train_pct,
            validate_pct=validate_pct,
            oos_pct=oos_pct,
            total_trades=len(sorted_trades),
        )

    # Welch's t-test (unequal variances, two-sided).
    # H0: mean(challenger_oos) == mean(champion_oos)
    # H1: mean(challenger_oos) != mean(champion_oos)
    t_stat, p_value = stats.ttest_ind(
        cl_oos_returns,
        ch_oos_returns,
        equal_var=False,
    )
    t_stat = float(t_stat)
    p_value = float(p_value)

    # --- Promotion decision ---
    # The Challenger is promoted only when ALL three conditions hold:
    #   1. OOS Sortino is strictly higher than Champion's.
    #   2. The improvement is statistically significant (p < α).
    #   3. Challenger t-statistic is positive (mean returns are higher).
    sortino_improvement = cl_oos.sortino_ratio > ch_oos.sortino_ratio
    significant = p_value < significance_level
    positive_direction = t_stat > 0

    promote = sortino_improvement and significant and positive_direction

    verdict = Verdict.PROMOTE if promote else Verdict.REJECT

    logger.info(
        "Walk-forward result: verdict=%s | "
        "challenger_oos_sortino=%.4f champion_oos_sortino=%.4f | "
        "t=%.4f p=%.6f (α=%.2f)",
        verdict.value,
        cl_oos.sortino_ratio,
        ch_oos.sortino_ratio,
        t_stat,
        p_value,
        significance_level,
    )

    return ValidationReport(
        verdict=verdict,
        promote_challenger=promote,
        t_statistic=round(t_stat, 6),
        p_value=round(p_value, 8),
        significance_level=significance_level,
        challenger=ParamSetReport(
            label="challenger",
            params=challenger_params,
            train=cl_train,
            validate_block=cl_val,
            oos=cl_oos,
        ),
        champion=ParamSetReport(
            label="champion",
            params=champion_params,
            train=ch_train,
            validate_block=ch_val,
            oos=ch_oos,
        ),
        train_pct=train_pct,
        validate_pct=validate_pct,
        oos_pct=oos_pct,
        total_trades=len(sorted_trades),
    )
