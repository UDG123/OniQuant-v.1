"""Walk-Forward Analysis for Champion vs. Challenger parameter validation.

Partitions ``ml_training_data`` into three sequential time-ordered
blocks — Train, Validate, Out-of-Sample — runs a vectorized simulation
for both parameter sets across all blocks, and applies a Welch t-test
on the OOS block to determine whether the Challenger's Sortino
improvement is statistically significant (p < 0.05).

The ``ValidationReport`` includes a full ``HypothesisTestProof`` with
the mathematical derivation: Welch-Satterthwaite degrees of freedom,
Cohen's d effect size, confidence interval, and the Sortino formula
used for evaluation.

References
----------
[16] Bailey, D. H. et al. "The Deflated Sharpe Ratio" (2014).
[21] Harvey, C. R. & Liu, Y. "Backtesting" (2015).

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
import math
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
    downside_deviation: float = Field(
        ..., description="Std-dev of negative returns (ddof=1)",
    )
    win_rate: float = Field(..., description="Fraction of accepted trades with pnl > 0")
    max_drawdown_pct: float = Field(
        ..., description="Maximum peak-to-trough drawdown on the equity curve",
    )


class SampleStatistics(BaseModel):
    """Descriptive statistics for one OOS sample — feeds the t-test proof."""

    label: str = Field(..., description="'champion' or 'challenger'")
    n: int = Field(..., description="Sample size (n_accepted in OOS)")
    mean: float = Field(..., description="Sample mean of OOS returns")
    std: float = Field(..., description="Sample std-dev (ddof=1)")
    variance: float = Field(..., description="s^2 = std^2")
    sortino_ratio: float = Field(..., description="Annualised Sortino on OOS")


class HypothesisTestProof(BaseModel):
    """Full mathematical proof of the Independent Welch t-test.

    Documents every intermediate value so the promotion decision is
    auditable and reproducible.

    Formulas
    --------
    Welch t-statistic::

        t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

    Welch-Satterthwaite degrees of freedom::

        df = (s1^2/n1 + s2^2/n2)^2
             / ( (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) )

    Cohen's d (pooled)::

        d = (x_bar_1 - x_bar_2) / s_pooled
        s_pooled = sqrt( ((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2) )

    Sortino ratio (annualised, MAR = 0)::

        Sortino = (mean_return / downside_std) * sqrt(252)

    Confidence interval for the mean difference::

        CI = (x_bar_1 - x_bar_2) +/- t_crit * SE
        SE = sqrt(s1^2/n1 + s2^2/n2)
    """

    null_hypothesis: str = Field(
        default=(
            "H0: mu_challenger = mu_champion — the mean OOS return of the "
            "Challenger equals the mean OOS return of the Champion."
        ),
    )
    alternative_hypothesis: str = Field(
        default=(
            "H1: mu_challenger != mu_champion — the mean OOS returns differ."
        ),
    )

    challenger_stats: SampleStatistics
    champion_stats: SampleStatistics

    # Welch t-test intermediates
    standard_error: float = Field(
        ..., description="SE = sqrt(s1^2/n1 + s2^2/n2)",
    )
    t_statistic: float = Field(
        ..., description="t = (x_bar_challenger - x_bar_champion) / SE",
    )
    degrees_of_freedom: float = Field(
        ..., description="Welch-Satterthwaite approximate df",
    )
    p_value: float = Field(..., description="Two-sided p-value from t(df)")
    significance_level: float = Field(
        default=_SIGNIFICANCE_LEVEL,
        description="Alpha threshold for rejection of H0",
    )
    reject_null: bool = Field(..., description="True when p < alpha")

    # Confidence interval on the mean difference
    mean_difference: float = Field(
        ..., description="x_bar_challenger - x_bar_champion",
    )
    ci_lower: float = Field(..., description="Lower bound of (1-alpha) CI")
    ci_upper: float = Field(..., description="Upper bound of (1-alpha) CI")

    # Effect size
    cohens_d: float = Field(
        ..., description="Cohen's d (pooled SD) — effect magnitude",
    )
    effect_interpretation: str = Field(
        ..., description="Negligible / Small / Medium / Large per Cohen (1988)",
    )

    # Sortino-specific gate
    sortino_improvement: bool = Field(
        ..., description="True when Challenger OOS Sortino > Champion OOS Sortino",
    )
    sortino_delta: float = Field(
        ..., description="Challenger Sortino - Champion Sortino on OOS",
    )

    # Final decision
    sortino_formula: str = Field(
        default="Sortino = (mean_return / downside_std) * sqrt(252)  [MAR = 0]",
    )
    promotion_gates: dict[str, bool] = Field(
        ...,
        description=(
            "All three gates must be True for promotion: "
            "sortino_improved, statistically_significant, positive_direction"
        ),
    )


class ParamSetReport(BaseModel):
    """Full walk-forward results for a single parameter set."""

    label: str = Field(..., description="'champion' or 'challenger'")
    params: dict[str, Any]
    train: BlockMetrics
    validate_block: BlockMetrics
    oos: BlockMetrics


class ValidationReport(BaseModel):
    """Complete Walk-Forward Analysis report comparing Champion vs. Challenger.

    Contains the ``proof`` field — a ``HypothesisTestProof`` with the
    full mathematical derivation of the statistical test, intermediate
    values, and promotion decision logic.
    """

    verdict: Verdict
    promote_challenger: bool = Field(
        ..., description="True when Challenger passes all gates",
    )
    proof: HypothesisTestProof = Field(
        ..., description="Full math proof of the Welch t-test and promotion logic",
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

    Matches the implementation in ``src.core.optimizer`` exactly::

        Sortino = (mean_return / downside_std) * sqrt(252)
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


def _downside_deviation(returns: np.ndarray) -> float:
    """Sample standard deviation of negative returns (ddof=1)."""
    downside = returns[returns < 0.0]
    if len(downside) < 2:
        return 0.0
    return float(np.std(downside, ddof=1))


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

    returns = (
        np.array(accepted_returns, dtype=np.float64)
        if accepted_returns
        else np.array([], dtype=np.float64)
    )
    n_accepted = len(returns)

    sortino = _compute_sortino(returns)
    dd = _downside_deviation(returns)
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
        downside_deviation=round(dd, 8),
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
# Math proof construction
# ---------------------------------------------------------------------------


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude per Cohen (1988) conventions."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    if abs_d < 0.5:
        return "Small"
    if abs_d < 0.8:
        return "Medium"
    return "Large"


def _build_proof(
    cl_oos_returns: np.ndarray,
    ch_oos_returns: np.ndarray,
    cl_oos_sortino: float,
    ch_oos_sortino: float,
    significance_level: float,
) -> HypothesisTestProof:
    """Construct the full hypothesis-test proof from OOS return arrays.

    Computes the Welch t-statistic, Welch-Satterthwaite degrees of
    freedom, confidence interval, and Cohen's d — all from first
    principles so every intermediate is auditable.
    """
    # --- Sample statistics ---
    n1 = len(cl_oos_returns)
    n2 = len(ch_oos_returns)
    x1 = float(np.mean(cl_oos_returns))
    x2 = float(np.mean(ch_oos_returns))
    s1 = float(np.std(cl_oos_returns, ddof=1))
    s2 = float(np.std(ch_oos_returns, ddof=1))
    v1 = s1 ** 2
    v2 = s2 ** 2

    # --- Welch t-statistic ---
    #   t = (x1 - x2) / sqrt(s1^2/n1 + s2^2/n2)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (x1 - x2) / se if se > 0 else 0.0

    # --- Welch-Satterthwaite degrees of freedom ---
    #   df = (s1^2/n1 + s2^2/n2)^2
    #        / ( (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) )
    numerator = (v1 / n1 + v2 / n2) ** 2
    denominator = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
    df = numerator / denominator if denominator > 0 else 1.0

    # --- p-value (two-sided) from Student t-distribution ---
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df))

    # --- Confidence interval for mean difference ---
    #   CI = (x1 - x2) +/- t_crit * SE
    t_crit = float(stats.t.ppf(1.0 - significance_level / 2.0, df))
    mean_diff = x1 - x2
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    # --- Cohen's d (pooled SD) ---
    #   s_pooled = sqrt( ((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2) )
    #   d = (x1 - x2) / s_pooled
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    s_pooled = math.sqrt(pooled_var) if pooled_var > 0 else 0.0
    cohens_d = mean_diff / s_pooled if s_pooled > 0 else 0.0

    # --- Gate evaluation ---
    reject_null = p_value < significance_level
    sortino_improved = cl_oos_sortino > ch_oos_sortino
    positive_direction = t_stat > 0

    challenger_sample = SampleStatistics(
        label="challenger",
        n=n1,
        mean=round(x1, 8),
        std=round(s1, 8),
        variance=round(v1, 10),
        sortino_ratio=round(cl_oos_sortino, 6),
    )
    champion_sample = SampleStatistics(
        label="champion",
        n=n2,
        mean=round(x2, 8),
        std=round(s2, 8),
        variance=round(v2, 10),
        sortino_ratio=round(ch_oos_sortino, 6),
    )

    return HypothesisTestProof(
        challenger_stats=challenger_sample,
        champion_stats=champion_sample,
        standard_error=round(se, 10),
        t_statistic=round(t_stat, 6),
        degrees_of_freedom=round(df, 4),
        p_value=round(p_value, 8),
        significance_level=significance_level,
        reject_null=reject_null,
        mean_difference=round(mean_diff, 8),
        ci_lower=round(ci_lower, 8),
        ci_upper=round(ci_upper, 8),
        cohens_d=round(cohens_d, 6),
        effect_interpretation=_interpret_cohens_d(cohens_d),
        sortino_improvement=sortino_improved,
        sortino_delta=round(cl_oos_sortino - ch_oos_sortino, 6),
        promotion_gates={
            "sortino_improved": sortino_improved,
            "statistically_significant": reject_null,
            "positive_direction": positive_direction,
        },
    )


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
        full hypothesis-test proof, and a promotion verdict.
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

    # --- Build proof (or stub for insufficient data) ---
    if len(ch_oos_returns) < _MIN_OOS_TRADES or len(cl_oos_returns) < _MIN_OOS_TRADES:
        logger.warning(
            "Insufficient OOS trades for t-test: champion=%d challenger=%d (min=%d)",
            len(ch_oos_returns),
            len(cl_oos_returns),
            _MIN_OOS_TRADES,
        )
        # Produce a stub proof with zeroed-out statistics.
        stub_cl = SampleStatistics(
            label="challenger",
            n=len(cl_oos_returns),
            mean=float(np.mean(cl_oos_returns)) if len(cl_oos_returns) > 0 else 0.0,
            std=float(np.std(cl_oos_returns, ddof=1)) if len(cl_oos_returns) > 1 else 0.0,
            variance=0.0,
            sortino_ratio=cl_oos.sortino_ratio,
        )
        stub_ch = SampleStatistics(
            label="champion",
            n=len(ch_oos_returns),
            mean=float(np.mean(ch_oos_returns)) if len(ch_oos_returns) > 0 else 0.0,
            std=float(np.std(ch_oos_returns, ddof=1)) if len(ch_oos_returns) > 1 else 0.0,
            variance=0.0,
            sortino_ratio=ch_oos.sortino_ratio,
        )
        proof = HypothesisTestProof(
            challenger_stats=stub_cl,
            champion_stats=stub_ch,
            standard_error=0.0,
            t_statistic=0.0,
            degrees_of_freedom=0.0,
            p_value=1.0,
            significance_level=significance_level,
            reject_null=False,
            mean_difference=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            cohens_d=0.0,
            effect_interpretation="Negligible",
            sortino_improvement=False,
            sortino_delta=0.0,
            promotion_gates={
                "sortino_improved": False,
                "statistically_significant": False,
                "positive_direction": False,
            },
        )
        return ValidationReport(
            verdict=Verdict.INSUFFICIENT_DATA,
            promote_challenger=False,
            proof=proof,
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

    # --- Full proof from first principles ---
    proof = _build_proof(
        cl_oos_returns,
        ch_oos_returns,
        cl_oos.sortino_ratio,
        ch_oos.sortino_ratio,
        significance_level,
    )

    # --- Promotion decision: all three gates must be True ---
    gates = proof.promotion_gates
    promote = all(gates.values())
    verdict = Verdict.PROMOTE if promote else Verdict.REJECT

    logger.info(
        "Walk-forward result: verdict=%s | "
        "challenger_oos_sortino=%.4f champion_oos_sortino=%.4f | "
        "t=%.4f p=%.6f df=%.1f cohen_d=%.4f (alpha=%.2f) | "
        "gates=%s",
        verdict.value,
        cl_oos.sortino_ratio,
        ch_oos.sortino_ratio,
        proof.t_statistic,
        proof.p_value,
        proof.degrees_of_freedom,
        proof.cohens_d,
        significance_level,
        gates,
    )

    return ValidationReport(
        verdict=verdict,
        promote_challenger=promote,
        proof=proof,
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
