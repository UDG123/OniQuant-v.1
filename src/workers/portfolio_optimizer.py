"""Meta-optimization job — Global Portfolio Sortino via Optuna + Kelly.

Celery Beat task (weekly) that optimises the capital-weight allocation
across all five trading desks by maximising the **Global Portfolio
Sortino Ratio** subject to Fractional Kelly constraints.

Pipeline:

1. Fetch 30-day realised P&L per desk from ``trade_log`` in PostgreSQL.
2. Optuna study maximises the portfolio Sortino by tuning per-desk
   ``capital_weight`` (0.0–1.0, sum normalised to 1.0).
3. Each candidate weight vector is clamped so no desk exceeds its
   Fractional Kelly ceiling (0.25 × f*), preventing over-leverage from
   any single desk's edge estimate.
4. The winning ``best_weights`` JSON is sent to ClaudeCTO for a
   "Regime-Alignment" semantic check before being hot-swapped into
   Redis.

Start the worker::

    celery -A src.workers.portfolio_optimizer worker --beat --loglevel=info
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import optuna
import orjson
from celery import Celery
from celery.schedules import crontab
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
)
SYNC_POSTGRES_URL: str = POSTGRES_URL.replace("+asyncpg", "")

CLAUDE_API_KEY: str | None = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

_LOOKBACK_DAYS: int = int(os.getenv("PORTFOLIO_LOOKBACK_DAYS", "30"))
_N_TRIALS: int = int(os.getenv("PORTFOLIO_N_TRIALS", "200"))
_STUDY_NAME: str = os.getenv("PORTFOLIO_STUDY_NAME", "oniquant_portfolio_weights")
_FRACTIONAL_KELLY: float = float(os.getenv("PORTFOLIO_KELLY_FRACTION", "0.25"))
_MIN_TRADES_PER_DESK: int = int(os.getenv("PORTFOLIO_MIN_TRADES", "10"))
_ANNUAL_TRADING_PERIODS: int = 252
_DESK_IDS: list[int] = [1, 2, 3, 4, 5]
_CTO_TIMEOUT: float = float(os.getenv("PORTFOLIO_CTO_TIMEOUT", "30.0"))

logger = logging.getLogger("oniquant.portfolio_optimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

celery_app = Celery(
    "portfolio_optimizer",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    beat_schedule={
        "weekly-portfolio-optimization": {
            "task": "src.workers.portfolio_optimizer.run_portfolio_optimization",
            "schedule": crontab(hour=5, minute=0, day_of_week="sunday"),
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

# ---------------------------------------------------------------------------
# SQL — per-desk realised P&L + Kelly inputs
# ---------------------------------------------------------------------------

_PNL_QUERY = text("""
    SELECT
        desk_id,
        pnl_pct,
        closed_at
    FROM trade_log
    WHERE status = 'SIM_CLOSED'
      AND closed_at >= NOW() - MAKE_INTERVAL(days => :lookback_days)
    ORDER BY desk_id, closed_at ASC
""")

_KELLY_QUERY = text("""
    SELECT
        desk_id,
        COUNT(*)                                              AS total_trades,
        SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)        AS winning_trades,
        AVG(CASE WHEN pnl_pct > 0 THEN pnl_pct END)         AS avg_win_pct,
        AVG(CASE WHEN pnl_pct <= 0 THEN ABS(pnl_pct) END)   AS avg_loss_pct
    FROM trade_log
    WHERE status = 'SIM_CLOSED'
      AND closed_at >= NOW() - MAKE_INTERVAL(days => :lookback_days)
    GROUP BY desk_id
    ORDER BY desk_id
""")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_desk_returns(
    lookback_days: int = _LOOKBACK_DAYS,
) -> dict[int, np.ndarray]:
    """Fetch per-desk return arrays from ``trade_log``."""
    db: Session = _SessionLocal()
    try:
        rows = db.execute(
            _PNL_QUERY, {"lookback_days": lookback_days},
        ).fetchall()
    finally:
        db.close()

    desk_returns: dict[int, list[float]] = {d: [] for d in _DESK_IDS}
    for row in rows:
        did = int(row.desk_id)
        if did in desk_returns:
            desk_returns[did].append(float(row.pnl_pct))

    return {
        d: np.array(rets, dtype=np.float64)
        for d, rets in desk_returns.items()
    }


def _load_kelly_ceilings(
    lookback_days: int = _LOOKBACK_DAYS,
) -> dict[int, float]:
    """Compute the Fractional Kelly ceiling for each desk.

    f* = (p · b − q) / b

    Returns 0.25 × f* per desk (clamped to [0, 1]).
    """
    db: Session = _SessionLocal()
    try:
        rows = db.execute(
            _KELLY_QUERY, {"lookback_days": lookback_days},
        ).fetchall()
    finally:
        db.close()

    ceilings: dict[int, float] = {d: 0.0 for d in _DESK_IDS}
    for row in rows:
        did = int(row.desk_id)
        total = int(row.total_trades)
        wins = int(row.winning_trades or 0)
        avg_win = float(row.avg_win_pct or 0.0)
        avg_loss = float(row.avg_loss_pct or 0.0)

        if total < _MIN_TRADES_PER_DESK or avg_loss <= 0:
            ceilings[did] = 0.0
            continue

        p = wins / total
        b = avg_win / avg_loss  # payout ratio
        q = 1.0 - p

        full_kelly = (p * b - q) / b if b > 0 else 0.0
        full_kelly = max(full_kelly, 0.0)

        # Fractional Kelly: 0.25 × f*
        ceilings[did] = min(full_kelly * _FRACTIONAL_KELLY, 1.0)

    return ceilings


# ---------------------------------------------------------------------------
# Sortino computation
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


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _objective(
    trial: optuna.Trial,
    desk_returns: dict[int, np.ndarray],
    kelly_ceilings: dict[int, float],
) -> float:
    """Maximise the Global Portfolio Sortino Ratio.

    For each desk, Optuna suggests a raw ``capital_weight`` ∈ [0, 1].
    The weights are then:
    1. Clamped to the desk's Fractional Kelly ceiling.
    2. Normalised to sum to 1.0 (portfolio constraint).

    The weighted return series is aggregated into a single portfolio
    return stream and scored via the annualised Sortino ratio.
    """
    raw_weights: dict[int, float] = {}
    for did in _DESK_IDS:
        raw_weights[did] = trial.suggest_float(
            f"w_desk{did}", 0.0, 1.0,
        )

    # --- Kelly ceiling clamp -----------------------------------------------
    clamped: dict[int, float] = {}
    for did in _DESK_IDS:
        ceiling = kelly_ceilings.get(did, 0.0)
        if ceiling <= 0 or len(desk_returns.get(did, [])) < _MIN_TRADES_PER_DESK:
            clamped[did] = 0.0
        else:
            clamped[did] = min(raw_weights[did], ceiling)

    # --- Normalise to sum=1 ------------------------------------------------
    total = sum(clamped.values())
    if total <= 0:
        return -10.0  # No eligible desk — penalise heavily.

    weights = {d: clamped[d] / total for d in _DESK_IDS}

    # --- Build weighted portfolio return stream ----------------------------
    # Align all desks to the same length via zero-padded arrays.
    max_len = max(
        (len(desk_returns[d]) for d in _DESK_IDS if len(desk_returns[d]) > 0),
        default=0,
    )
    if max_len < 2:
        return -10.0

    portfolio_returns = np.zeros(max_len, dtype=np.float64)

    for did in _DESK_IDS:
        rets = desk_returns[did]
        if len(rets) == 0 or weights[did] == 0:
            continue
        padded = np.zeros(max_len, dtype=np.float64)
        padded[: len(rets)] = rets
        portfolio_returns += weights[did] * padded

    sortino = _compute_sortino(portfolio_returns)

    # Report per-desk weights for dashboard monitoring.
    for did in _DESK_IDS:
        trial.set_user_attr(f"weight_desk{did}", round(weights[did], 6))
    trial.set_user_attr("portfolio_sortino", round(sortino, 6))

    return sortino


# ---------------------------------------------------------------------------
# ClaudeCTO regime-alignment check (sync, for Celery worker context)
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_cto_response(raw_text: str) -> tuple[float, str]:
    """Extract consensus_score and reasoning from LLM text."""
    parsed: dict | None = None
    try:
        parsed = orjson.loads(raw_text)
    except (orjson.JSONDecodeError, ValueError):
        pass

    if parsed is None:
        match = _JSON_BLOCK_RE.search(raw_text)
        if match is None:
            raise ValueError(f"No JSON found in CTO response: {raw_text!r}")
        parsed = orjson.loads(match.group())

    score = float(parsed.get("consensus_score", 0))
    reasoning = str(
        parsed.get("reasoning_summary")
        or parsed.get("reasoning")
        or parsed.get("summary")
        or ""
    ).strip()

    return score, reasoning


def _regime_alignment_check(
    best_weights: dict[str, float],
    kelly_ceilings: dict[int, float],
    portfolio_sortino: float,
) -> dict[str, Any]:
    """Send the optimised weights to ClaudeCTO for regime-alignment review.

    Uses a synchronous ``httpx`` call (this runs inside a Celery worker,
    not the async FastAPI event loop).

    Returns
    -------
    dict
        Keys: ``approved`` (bool), ``score`` (float), ``reasoning`` (str).
    """
    if not CLAUDE_API_KEY:
        logger.warning("CLAUDE_API_KEY not set — skipping CTO regime check")
        return {
            "approved": False,
            "score": 0.0,
            "reasoning": "API key not configured",
        }

    system_prompt = (
        "You are a portfolio risk officer reviewing proposed capital weight "
        "changes across 5 trading desks. Evaluate whether the weights are "
        "aligned with the current market regime. Consider concentration risk, "
        "desk-level Kelly ceilings, and whether the overall portfolio Sortino "
        "ratio improvement justifies the rebalance. "
        'Output JSON: {"consensus_score": 1-10, "reasoning_summary": "..."}. '
        "Score >= 7 means APPROVED for production deployment."
    )

    payload = {
        "proposed_weights": best_weights,
        "kelly_ceilings": {f"desk{k}": round(v, 4) for k, v in kelly_ceilings.items()},
        "portfolio_sortino": round(portfolio_sortino, 4),
        "optimization_window_days": _LOOKBACK_DAYS,
        "fractional_kelly_factor": _FRACTIONAL_KELLY,
    }

    api_body = orjson.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 256,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": json.dumps(payload, indent=2),
            },
        ],
    })

    try:
        with httpx.Client(timeout=_CTO_TIMEOUT) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": CLAUDE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                content=api_body,
            )

        if response.status_code != 200:
            logger.error(
                "CTO API error %d: %s", response.status_code, response.text,
            )
            return {
                "approved": False,
                "score": 0.0,
                "reasoning": f"API error {response.status_code}",
            }

        body = response.json()
        raw_text: str = body.get("content", [{}])[0].get("text", "")

        score, reasoning = _parse_cto_response(raw_text)
        approved = score >= 7.0

        logger.info(
            "CTO regime check: score=%.1f %s — %s",
            score,
            "APPROVED" if approved else "REJECTED",
            reasoning,
        )

        return {
            "approved": approved,
            "score": score,
            "reasoning": reasoning,
        }

    except Exception:
        logger.exception("CTO regime-alignment check failed")
        return {
            "approved": False,
            "score": 0.0,
            "reasoning": "Exception during CTO dispatch",
        }


# ---------------------------------------------------------------------------
# Redis weight update (sync, for Celery worker context)
# ---------------------------------------------------------------------------


def _update_weights_in_redis(weights: dict[str, float]) -> None:
    """Write the approved portfolio weights to Redis.

    Stores the weight vector under ``portfolio:weights`` as a JSON hash
    and publishes a notification on the strategy updates channel.
    """
    import redis

    r = redis.from_url(REDIS_URL, decode_responses=True)

    weight_payload = json.dumps({
        "weights": weights,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "fractional_kelly": _FRACTIONAL_KELLY,
        "lookback_days": _LOOKBACK_DAYS,
    })

    pipe = r.pipeline(transaction=True)
    pipe.set("portfolio:weights", weight_payload)
    # Also write per-desk keys for fast individual lookups.
    for key, value in weights.items():
        desk_num = key.replace("desk", "")
        pipe.hset(f"desk:{desk_num}:state", "capital_weight", str(value))
    pipe.publish(
        "oniquant:strategy_updates",
        json.dumps({
            "event": "PORTFOLIO_REBALANCE",
            "weights": weights,
            "timestamp": time.time(),
        }),
    )
    pipe.execute()

    logger.info("Portfolio weights updated in Redis: %s", weights)


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------


@celery_app.task(
    name="src.workers.portfolio_optimizer.run_portfolio_optimization",
    bind=True,
    max_retries=1,
    default_retry_delay=300,
)
def run_portfolio_optimization(self) -> dict[str, Any]:
    """Execute one round of portfolio weight optimisation.

    Steps:
    1. Fetch 30-day realised P&L and Kelly ceilings per desk.
    2. Optuna maximises the Global Portfolio Sortino by tweaking
       per-desk capital_weight [0, 1], clamped to Kelly ceilings.
    3. Best weights sent to ClaudeCTO for regime-alignment check.
    4. If approved (score >= 7.0), weights are hot-swapped into Redis.
    """
    try:
        # --- Step 1: Load data ------------------------------------------------
        desk_returns = _load_desk_returns(_LOOKBACK_DAYS)
        kelly_ceilings = _load_kelly_ceilings(_LOOKBACK_DAYS)

        active_desks = [
            d for d in _DESK_IDS
            if len(desk_returns.get(d, [])) >= _MIN_TRADES_PER_DESK
        ]

        if len(active_desks) < 2:
            logger.warning(
                "Only %d desk(s) have >= %d trades — skipping optimisation",
                len(active_desks), _MIN_TRADES_PER_DESK,
            )
            return {
                "status": "skipped",
                "reason": "insufficient_desks",
                "active_desks": active_desks,
            }

        logger.info(
            "Portfolio optimisation starting: %d desks, %d-day window, "
            "%d trials | Kelly ceilings: %s",
            len(active_desks),
            _LOOKBACK_DAYS,
            _N_TRIALS,
            {d: round(kelly_ceilings[d], 4) for d in _DESK_IDS},
        )

        # --- Step 2: Optuna study ---------------------------------------------
        storage = optuna.storages.RDBStorage(
            url=SYNC_POSTGRES_URL,
            engine_kwargs={"pool_pre_ping": True, "pool_size": 3},
        )

        study = optuna.create_study(
            study_name=_STUDY_NAME,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
            load_if_exists=True,
        )

        study.optimize(
            lambda trial: _objective(trial, desk_returns, kelly_ceilings),
            n_trials=_N_TRIALS,
        )

        best = study.best_trial
        best_weights: dict[str, float] = {
            f"desk{d}": round(best.user_attrs.get(f"weight_desk{d}", 0.0), 6)
            for d in _DESK_IDS
        }
        portfolio_sortino = best.user_attrs.get("portfolio_sortino", best.value)

        logger.info(
            "Optimisation complete — best Sortino=%.4f (trial #%d)\n"
            "  weights=%s",
            best.value,
            best.number,
            best_weights,
        )

        # --- Step 3: ClaudeCTO regime-alignment check -------------------------
        cto_result = _regime_alignment_check(
            best_weights, kelly_ceilings, portfolio_sortino,
        )

        result: dict[str, Any] = {
            "status": "completed",
            "best_trial": best.number,
            "portfolio_sortino": round(best.value, 6),
            "best_weights": best_weights,
            "kelly_ceilings": {
                f"desk{d}": round(kelly_ceilings[d], 6) for d in _DESK_IDS
            },
            "cto_check": cto_result,
            "n_trials": _N_TRIALS,
            "lookback_days": _LOOKBACK_DAYS,
            "active_desks": active_desks,
        }

        # --- Step 4: Update Redis if approved ---------------------------------
        if cto_result.get("approved"):
            _update_weights_in_redis(best_weights)
            result["weights_deployed"] = True
            logger.info("Portfolio weights DEPLOYED to production")
        else:
            result["weights_deployed"] = False
            logger.warning(
                "Portfolio weights NOT deployed — CTO score=%.1f (%s)",
                cto_result.get("score", 0),
                cto_result.get("reasoning", "no reason"),
            )

        return result

    except Exception as exc:
        logger.exception("Portfolio optimisation failed")
        raise self.retry(exc=exc)
