"""ClaudeCTO reasoning-layer dispatcher.

Sends hydrated signal payloads to the Anthropic Messages API
(Claude 3.5 Sonnet), parses a strict ``consensus_score`` +
``reasoning_summary`` response, and logs every reasoning trace to
the ``reasoning_logs`` PostgreSQL table for future Test-Time
Reinforcement Learning (TTRL) dataset construction.

Usage::

    from src.services.claude_cto_dispatcher import dispatch_to_claude_cto

    result = await dispatch_to_claude_cto(hydrated_bytes, db_session)
    # result.status  -> "APPROVED" | "REJECTED"
    # result.score   -> 1.0–10.0
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import orjson
from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://user:password@localhost:5432/oniquant",
)
CLAUDE_API_KEY: str | None = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
CONSENSUS_THRESHOLD: float = float(os.getenv("CONSENSUS_THRESHOLD", "7.0"))
_API_TIMEOUT: float = float(os.getenv("CLAUDE_API_TIMEOUT", "30.0"))

logger = logging.getLogger("oniquant.cto_dispatcher")

# ---------------------------------------------------------------------------
# Async database engine (module-level singleton)
# ---------------------------------------------------------------------------

_engine = create_async_engine(POSTGRES_URL, pool_pre_ping=True, pool_size=5)
_async_session_factory = async_sessionmaker(
    _engine, class_=AsyncSession, expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# ORM — reasoning_logs table (TTRL dataset)
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


class ReasoningLog(_Base):
    """Persists every ClaudeCTO reasoning trace for TTRL training.

    Each row captures the full request context, the raw LLM response,
    the parsed score + summary, and the gate decision so downstream
    reinforcement-learning pipelines can reconstruct the decision
    boundary at any point in time.
    """

    __tablename__ = "reasoning_logs"
    __table_args__ = (
        Index("ix_reasoning_logs_created_at", "created_at"),
        Index("ix_reasoning_logs_signal_symbol", "signal_id", "symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Signal context
    signal_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    desk_id: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String(8), nullable=False)

    # LLM request / response
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    hydrated_payload: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response: Mapped[str] = mapped_column(Text, nullable=False)

    # Parsed result
    consensus_score: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning_summary: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="APPROVED or REJECTED",
    )

    # Token usage
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Latency
    latency_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Round-trip API latency in milliseconds",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CTODispatchResult:
    """Immutable result returned by ``dispatch_to_claude_cto``."""

    consensus_score: float
    reasoning_summary: str
    status: str  # "APPROVED" | "REJECTED"
    raw_response: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


# ---------------------------------------------------------------------------
# Shared httpx client (lazy singleton, mirrors webhooks.py pattern)
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


async def _get_http_client() -> httpx.AsyncClient:
    """Lazy-initialised shared ``httpx.AsyncClient``."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=_API_TIMEOUT)
    return _http_client


# ---------------------------------------------------------------------------
# Strict response parser
# ---------------------------------------------------------------------------

# Matches a JSON object with consensus_score and reasoning keys,
# tolerant of whitespace / markdown fences the LLM might emit.
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_cto_response(raw_text: str) -> tuple[float, str]:
    """Extract ``consensus_score`` and ``reasoning_summary`` from LLM text.

    Strategy:
        1. Try ``orjson.loads`` on the raw text (fast path for clean JSON).
        2. Fall back to regex extraction of the first ``{…}`` block.
        3. Validate score is a number within [1.0, 10.0].

    Returns
    -------
    tuple[float, str]
        (consensus_score, reasoning_summary)

    Raises
    ------
    ValueError
        If the response cannot be parsed or the score is out of range.
    """
    parsed: dict | None = None

    # Fast path: entire response is valid JSON.
    try:
        parsed = orjson.loads(raw_text)
    except (orjson.JSONDecodeError, ValueError):
        pass

    # Fallback: extract first JSON object from markdown / fenced output.
    if parsed is None:
        match = _JSON_BLOCK_RE.search(raw_text)
        if match is None:
            raise ValueError(f"No JSON object found in LLM response: {raw_text!r}")
        try:
            parsed = orjson.loads(match.group())
        except (orjson.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Extracted block is not valid JSON: {match.group()!r}"
            ) from exc

    # --- Extract & validate consensus_score ---
    raw_score = parsed.get("consensus_score")
    if raw_score is None:
        raise ValueError(f"'consensus_score' key missing from response: {parsed}")

    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'consensus_score' is not numeric: {raw_score!r}"
        ) from exc

    if not (1.0 <= score <= 10.0):
        raise ValueError(
            f"consensus_score {score} outside valid range [1.0, 10.0]"
        )

    # --- Extract reasoning_summary ---
    reasoning = str(
        parsed.get("reasoning_summary")
        or parsed.get("reasoning")
        or parsed.get("summary")
        or ""
    ).strip()

    if not reasoning:
        raise ValueError(
            f"No reasoning_summary / reasoning key found in response: {parsed}"
        )

    return score, reasoning


# ---------------------------------------------------------------------------
# Main dispatch function
# ---------------------------------------------------------------------------


async def dispatch_to_claude_cto(
    hydrated_payload: bytes,
    db: AsyncSession | None = None,
) -> CTODispatchResult:
    """Send a hydrated signal to the Anthropic Messages API and log the trace.

    Parameters
    ----------
    hydrated_payload : bytes
        orjson-serialised payload produced by ``signal_hydration.hydrate_signal()``.
    db : AsyncSession | None
        Optional async session.  When provided the reasoning trace is
        persisted to ``reasoning_logs``.  When ``None`` a throwaway
        session is created from the module-level factory.

    Returns
    -------
    CTODispatchResult
        Parsed and gate-evaluated result with APPROVED / REJECTED status.

    Raises
    ------
    RuntimeError
        If ``CLAUDE_API_KEY`` is not configured.
    httpx.HTTPStatusError
        On non-200 API responses (after logging).
    ValueError
        If the LLM response cannot be parsed.
    """
    if not CLAUDE_API_KEY:
        raise RuntimeError(
            "CLAUDE_API_KEY environment variable is not set — "
            "cannot dispatch to ClaudeCTO"
        )

    # Deserialise hydrated bytes to split system_prompt from user content.
    payload: dict = orjson.loads(hydrated_payload)
    system_prompt: str = payload.pop("system_prompt", "")

    signal_id: str = payload.get("webhook", {}).get("signal_id", "unknown")
    symbol: str = payload.get("webhook", {}).get("symbol", "unknown")
    desk_id: int = payload.get("desk_state", {}).get("desk_id", 0)
    action: str = payload.get("webhook", {}).get("action", "unknown")

    # Build the Messages API request body.
    api_body = orjson.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 256,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": orjson.dumps(payload).decode(),
            },
        ],
    })

    client = await _get_http_client()
    start_ns = _monotonic_ns()

    response = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        content=api_body,
    )

    latency_ms = (_monotonic_ns() - start_ns) // 1_000_000

    if response.status_code != 200:
        logger.error(
            "Anthropic API error %d for signal %s: %s",
            response.status_code,
            signal_id,
            response.text,
        )
        response.raise_for_status()

    body = response.json()
    raw_text: str = body.get("content", [{}])[0].get("text", "")
    usage = body.get("usage", {})
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    # --- Strict parse ---
    score, reasoning = _parse_cto_response(raw_text)
    gate_status = "APPROVED" if score >= CONSENSUS_THRESHOLD else "REJECTED"

    logger.info(
        "ClaudeCTO [%s] signal=%s symbol=%s score=%.1f → %s (%dms)",
        CLAUDE_MODEL,
        signal_id,
        symbol,
        score,
        gate_status,
        latency_ms,
    )

    # --- Persist reasoning trace to PostgreSQL for TTRL ---
    log_entry = ReasoningLog(
        signal_id=signal_id,
        symbol=symbol,
        desk_id=desk_id,
        action=action,
        model=CLAUDE_MODEL,
        hydrated_payload=hydrated_payload.decode(),
        raw_response=raw_text,
        consensus_score=score,
        reasoning_summary=reasoning,
        status=gate_status,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )

    own_session = False
    if db is None:
        db = _async_session_factory()
        own_session = True

    try:
        db.add(log_entry)
        await db.commit()
        logger.debug("Reasoning trace persisted for signal %s", signal_id)
    except Exception:
        await db.rollback()
        logger.exception("Failed to persist reasoning trace for signal %s", signal_id)
    finally:
        if own_session:
            await db.close()

    return CTODispatchResult(
        consensus_score=score,
        reasoning_summary=reasoning,
        status=gate_status,
        raw_response=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monotonic_ns() -> int:
    """Monotonic clock in nanoseconds (avoids importing time at module level)."""
    import time
    return time.monotonic_ns()
