"""Constitutional Guardrail Manager — self-correcting prompt addendum service.

Periodically audits the last 48 hours of ``trade_log`` and
``reasoning_logs`` to detect **Systemic Drifts**: cases where the
ClaudeCTO LLM issued high consensus scores that still resulted in
net-negative P&L on one or more trading desks.

When drifts are detected a cheaper, faster LLM (Claude Haiku) distils
the failure patterns into a strict 3-sentence "Constitutional Guardrail"
addendum.  This addendum is injected into the signal-hydration template
so the primary ClaudeCTO agent sees it as a "Current Market Context"
rule on every subsequent evaluation.

Lifecycle
---------
1. ``run_audit_loop()`` — long-running ``asyncio`` coroutine, call once
   at application startup via ``asyncio.create_task()``.
2. ``get_current_addendum()`` — synchronous read of the latest cached
   guardrail string; called from ``signal_hydration.hydrate_signal()``
   on every signal.

Start-up integration (e.g. in ``main.py``)::

    from src.services.constitutional_manager import run_audit_loop
    asyncio.create_task(run_audit_loop())
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import orjson
from sqlalchemy import Float, and_, cast, func, select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.models import TradeLog
from src.services.claude_cto_dispatcher import ReasoningLog

logger = logging.getLogger("oniquant.constitutional_manager")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://user:password@localhost:5432/oniquant",
)
CLAUDE_API_KEY: str | None = os.getenv("CLAUDE_API_KEY")

# Haiku is the cheapest/fastest model — ideal for meta-summarisation.
_GUARDRAIL_MODEL: str = os.getenv(
    "CONSTITUTIONAL_MODEL", "claude-haiku-4-5-20251001",
)
_GUARDRAIL_API_TIMEOUT: float = float(
    os.getenv("CONSTITUTIONAL_API_TIMEOUT", "20.0"),
)

# How far back to look for systemic drift evidence.
_LOOKBACK_HOURS: int = int(os.getenv("CONSTITUTIONAL_LOOKBACK_HOURS", "48"))

# How often the audit loop runs (default: every 30 minutes).
_AUDIT_INTERVAL_SECONDS: int = int(
    os.getenv("CONSTITUTIONAL_AUDIT_INTERVAL", "1800"),
)

# A consensus score is considered "high" if it meets or exceeds this
# threshold — matching the gate in claude_cto_dispatcher.py.
_HIGH_CONSENSUS_THRESHOLD: float = float(
    os.getenv("CONSENSUS_THRESHOLD", "7.0"),
)

# Minimum number of losing trades to trigger guardrail generation.
_MIN_DRIFT_TRADES: int = int(os.getenv("CONSTITUTIONAL_MIN_TRADES", "3"))

# ---------------------------------------------------------------------------
# Async DB engine
# ---------------------------------------------------------------------------

_engine = create_async_engine(POSTGRES_URL, pool_pre_ping=True, pool_size=3)
_session_factory = async_sessionmaker(
    _engine, class_=AsyncSession, expire_on_commit=False,
)

# ---------------------------------------------------------------------------
# HTTP client for Haiku calls
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=_GUARDRAIL_API_TIMEOUT)
    return _http_client


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class DeskDrift:
    """Aggregated drift evidence for a single trading desk."""

    desk_id: int
    symbol: str
    losing_trade_count: int
    avg_consensus_score: float
    avg_pnl_pct: float
    worst_pnl_pct: float
    sample_reasonings: list[str]


# ---------------------------------------------------------------------------
# Cached addendum — module-level singleton
# ---------------------------------------------------------------------------

_current_addendum: str = ""
_addendum_generated_at: float = 0.0


def get_current_addendum() -> str:
    """Return the latest Constitutional Guardrail addendum.

    This is a **synchronous** read of the module-level cache — safe to
    call from any async or sync context without awaiting.  Returns an
    empty string when no drift has been detected yet.
    """
    return _current_addendum


def get_addendum_age_seconds() -> float:
    """Seconds since the addendum was last refreshed (0.0 if never)."""
    if _addendum_generated_at == 0.0:
        return 0.0
    return time.time() - _addendum_generated_at


# ---------------------------------------------------------------------------
# Phase 1 — Query trade_log + reasoning_logs for systemic drifts
# ---------------------------------------------------------------------------


async def _identify_systemic_drifts(
    db: AsyncSession,
    lookback_hours: int = _LOOKBACK_HOURS,
) -> list[DeskDrift]:
    """Find desk/symbol pairs where high-consensus approvals led to losses.

    Joins ``trade_log`` (closed, losing trades) with ``reasoning_logs``
    (APPROVED, high-score entries) on ``desk_id``, ``symbol``, and a
    ±5-minute time window between the reasoning timestamp and the trade
    open timestamp.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    # Sub-query: losing closed trades in the lookback window.
    losing_trades = (
        select(
            TradeLog.desk_id,
            TradeLog.symbol,
            TradeLog.pnl_pct,
            TradeLog.opened_at,
        )
        .where(
            and_(
                TradeLog.status == "SIM_CLOSED",
                TradeLog.pnl_pct < 0,
                TradeLog.closed_at >= cutoff,
            )
        )
        .subquery("lt")
    )

    # Main query: join with reasoning_logs where the LLM approved with
    # a high consensus score within 5 minutes of the trade opening.
    stmt = (
        select(
            losing_trades.c.desk_id,
            losing_trades.c.symbol,
            func.count().label("trade_count"),
            func.avg(ReasoningLog.consensus_score).label("avg_score"),
            func.avg(cast(losing_trades.c.pnl_pct, Float)).label("avg_pnl"),
            func.min(cast(losing_trades.c.pnl_pct, Float)).label("worst_pnl"),
        )
        .join(
            ReasoningLog,
            and_(
                ReasoningLog.desk_id == losing_trades.c.desk_id,
                ReasoningLog.symbol == losing_trades.c.symbol,
                ReasoningLog.status == "APPROVED",
                ReasoningLog.consensus_score >= _HIGH_CONSENSUS_THRESHOLD,
                ReasoningLog.created_at >= cutoff,
                # Reasoning must precede the trade open by at most 5 min.
                func.abs(
                    func.extract(
                        "epoch",
                        ReasoningLog.created_at - losing_trades.c.opened_at,
                    )
                )
                <= 300,
            ),
        )
        .group_by(losing_trades.c.desk_id, losing_trades.c.symbol)
        .having(func.count() >= _MIN_DRIFT_TRADES)
        .order_by(func.avg(cast(losing_trades.c.pnl_pct, Float)).asc())
    )

    rows = (await db.execute(stmt)).all()

    if not rows:
        return []

    drifts: list[DeskDrift] = []

    for row in rows:
        # Fetch a small sample of the reasoning summaries for context.
        sample_stmt = (
            select(ReasoningLog.reasoning_summary)
            .where(
                and_(
                    ReasoningLog.desk_id == row.desk_id,
                    ReasoningLog.symbol == row.symbol,
                    ReasoningLog.status == "APPROVED",
                    ReasoningLog.consensus_score >= _HIGH_CONSENSUS_THRESHOLD,
                    ReasoningLog.created_at >= cutoff,
                )
            )
            .order_by(ReasoningLog.created_at.desc())
            .limit(5)
        )
        sample_rows = (await db.execute(sample_stmt)).scalars().all()

        drifts.append(
            DeskDrift(
                desk_id=row.desk_id,
                symbol=row.symbol,
                losing_trade_count=row.trade_count,
                avg_consensus_score=round(float(row.avg_score), 2),
                avg_pnl_pct=round(float(row.avg_pnl), 4),
                worst_pnl_pct=round(float(row.worst_pnl), 4),
                sample_reasonings=list(sample_rows),
            )
        )

    logger.info(
        "Identified %d systemic drift(s) across %d desk/symbol pairs",
        sum(d.losing_trade_count for d in drifts),
        len(drifts),
    )
    return drifts


# ---------------------------------------------------------------------------
# Phase 2 — Generate guardrail via Claude Haiku
# ---------------------------------------------------------------------------

_GUARDRAIL_SYSTEM_PROMPT = (
    "You are a risk-management auditor for an automated trading system. "
    "Given a summary of recent trading failures where the AI reasoner "
    "gave high confidence scores that still resulted in losses, produce "
    "a strict Constitutional Guardrail. "
    "Your output MUST be EXACTLY 3 sentences. "
    "Sentence 1: State the specific market condition or pattern that "
    "caused the overconfidence. "
    "Sentence 2: Name the desks and symbols affected and the magnitude "
    "of the average loss. "
    "Sentence 3: Issue a concrete directive the primary AI agent must "
    "follow to avoid repeating this failure. "
    "Do NOT use bullet points, headers, or any formatting. "
    "Output only the 3 sentences as plain text."
)


def _build_drift_summary(drifts: list[DeskDrift]) -> str:
    """Render drift evidence into a concise text block for the Haiku prompt."""
    lines: list[str] = []
    for d in drifts:
        lines.append(
            f"Desk {d.desk_id} / {d.symbol}: "
            f"{d.losing_trade_count} losing trades, "
            f"avg consensus score {d.avg_consensus_score}, "
            f"avg P&L {d.avg_pnl_pct:+.2%}, "
            f"worst P&L {d.worst_pnl_pct:+.2%}."
        )
        if d.sample_reasonings:
            lines.append(
                f"  Sample LLM reasonings: "
                + " | ".join(d.sample_reasonings[:3])
            )
    return "\n".join(lines)


async def generate_prompt_addendum(
    drifts: list[DeskDrift],
) -> str:
    """Call Claude Haiku to distil drift evidence into a 3-sentence guardrail.

    Parameters
    ----------
    drifts : list[DeskDrift]
        Aggregated drift evidence from ``_identify_systemic_drifts()``.

    Returns
    -------
    str
        A 3-sentence Constitutional Guardrail addendum, or an empty
        string if the API call fails (the system degrades gracefully).
    """
    if not CLAUDE_API_KEY:
        logger.warning(
            "CLAUDE_API_KEY not set — cannot generate constitutional guardrail"
        )
        return ""

    drift_summary = _build_drift_summary(drifts)

    api_body = orjson.dumps({
        "model": _GUARDRAIL_MODEL,
        "max_tokens": 300,
        "system": _GUARDRAIL_SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Here are the systemic drift failures from the last "
                    f"{_LOOKBACK_HOURS} hours:\n\n{drift_summary}"
                ),
            },
        ],
    })

    client = await _get_http_client()

    try:
        response = await client.post(
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
                "Haiku guardrail API error %d: %s",
                response.status_code,
                response.text,
            )
            return ""

        body = response.json()
        addendum: str = body.get("content", [{}])[0].get("text", "").strip()

        if not addendum:
            logger.warning("Haiku returned empty guardrail text")
            return ""

        logger.info(
            "Constitutional guardrail generated (%d chars, model=%s)",
            len(addendum),
            _GUARDRAIL_MODEL,
        )
        return addendum

    except httpx.HTTPError:
        logger.exception("Failed to call Haiku for guardrail generation")
        return ""


# ---------------------------------------------------------------------------
# Phase 3 — Periodic audit loop
# ---------------------------------------------------------------------------


async def run_audit_loop() -> None:
    """Long-running coroutine that periodically refreshes the guardrail.

    Steps per cycle:
        1. Query ``trade_log`` + ``reasoning_logs`` for the last 48 h.
        2. If systemic drifts are found, call ``generate_prompt_addendum()``.
        3. Cache the result so ``get_current_addendum()`` returns it.
        4. If no drifts are found, clear the cached addendum.
        5. Sleep for ``_AUDIT_INTERVAL_SECONDS`` and repeat.

    Intended to be started once during application lifespan::

        asyncio.create_task(run_audit_loop())
    """
    global _current_addendum, _addendum_generated_at

    logger.info(
        "Constitutional audit loop started — interval=%ds lookback=%dh",
        _AUDIT_INTERVAL_SECONDS,
        _LOOKBACK_HOURS,
    )

    while True:
        try:
            async with _session_factory() as db:
                drifts = await _identify_systemic_drifts(db)

            if drifts:
                addendum = await generate_prompt_addendum(drifts)
                if addendum:
                    _current_addendum = addendum
                    _addendum_generated_at = time.time()
                    logger.info(
                        "Guardrail cached — %d drift(s) summarised",
                        len(drifts),
                    )
                else:
                    logger.warning(
                        "Drifts detected but guardrail generation failed "
                        "— retaining previous addendum"
                    )
            else:
                if _current_addendum:
                    logger.info(
                        "No systemic drifts detected — clearing guardrail"
                    )
                _current_addendum = ""
                _addendum_generated_at = 0.0

        except Exception:
            logger.exception("Constitutional audit cycle failed")

        await asyncio.sleep(_AUDIT_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# One-shot manual trigger (useful for CLI / testing)
# ---------------------------------------------------------------------------


async def refresh_guardrail_now() -> str:
    """Run a single audit cycle immediately and return the guardrail text.

    Convenience wrapper for manual invocation or integration tests.
    """
    global _current_addendum, _addendum_generated_at

    async with _session_factory() as db:
        drifts = await _identify_systemic_drifts(db)

    if not drifts:
        _current_addendum = ""
        _addendum_generated_at = 0.0
        return ""

    addendum = await generate_prompt_addendum(drifts)
    if addendum:
        _current_addendum = addendum
        _addendum_generated_at = time.time()

    return _current_addendum
