"""TradingView webhook router — main signal ingestion entry point.

Receives alerts from TradingView, deduplicates via Redis, checks the
global volatility lock, hydrates the payload with desk state, and
dispatches it to the ClaudeCTO agent for a Consensus Score evaluation.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx
import orjson
from fastapi import APIRouter, Request, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.services.redis_manager import RedisStateManager
from src.services.signal_hydration import HydrationError, hydrate_signal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://user:password@localhost:5432/oniquant",
)
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

logger = logging.getLogger("oniquant.webhooks")

# ---------------------------------------------------------------------------
# Async database engine (shared across requests)
# ---------------------------------------------------------------------------

_async_engine = create_async_engine(POSTGRES_URL, pool_pre_ping=True, pool_size=5)
_async_session = async_sessionmaker(_async_engine, expire_on_commit=False)


class _Base(DeclarativeBase):
    pass


class VetoLog(_Base):
    """Persists every signal rejected by the global volatility lock."""

    __tablename__ = "veto_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(128), nullable=False, index=True)
    symbol = Column(String(32), nullable=False)
    reason = Column(String(64), nullable=False, default="volatility_lock")
    raw_payload = Column(Text, nullable=True)
    vetoed_at = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------


class TradingViewAlert(BaseModel):
    """Schema for incoming TradingView webhook payloads."""

    signal_id: str = Field(..., min_length=1, description="Unique alert identifier")
    symbol: str = Field(..., min_length=1, description="Instrument (e.g. XAUUSD)")
    action: str = Field(..., pattern=r"^(BUY|SELL|CLOSE)$", description="Trade action")
    desk_id: int = Field(default=1, ge=1, description="Target trading desk")
    price: float | None = Field(default=None, gt=0, description="Alert trigger price")
    timeframe: str | None = Field(default=None, description="Chart timeframe")
    message: str | None = Field(default=None, description="Free-text context")


# ---------------------------------------------------------------------------
# ClaudeCTO dispatch
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


async def _dispatch_to_claude_cto(hydrated_payload: bytes) -> dict[str, Any]:
    """Send the hydrated signal to the Claude API for Consensus Score evaluation.

    Returns the parsed response body or an error dict on failure.
    """
    if not CLAUDE_API_KEY:
        logger.warning("CLAUDE_API_KEY not set — skipping CTO dispatch")
        return {"consensus_score": None, "reasoning": "API key not configured"}

    payload = orjson.loads(hydrated_payload)
    system_prompt = payload.pop("system_prompt", "")

    client = await _get_http_client()
    response = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        content=orjson.dumps(
            {
                "model": CLAUDE_MODEL,
                "max_tokens": 256,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": orjson.dumps(payload).decode(),
                    }
                ],
            }
        ),
    )

    if response.status_code != 200:
        logger.error("Claude API error %d: %s", response.status_code, response.text)
        return {"consensus_score": None, "reasoning": f"API error {response.status_code}"}

    body = response.json()
    text = body.get("content", [{}])[0].get("text", "")

    # Attempt to parse structured JSON from the model response
    try:
        return orjson.loads(text)
    except (orjson.JSONDecodeError, ValueError):
        return {"consensus_score": None, "reasoning": text}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1", tags=["webhooks"])


@router.post(
    "/tradingview-alert",
    status_code=status.HTTP_200_OK,
    response_class=ORJSONResponse,
)
async def tradingview_alert(alert: TradingViewAlert, request: Request):
    """Ingest a TradingView webhook alert through the full signal pipeline.

    Pipeline stages:
    1. Pydantic validation (automatic via ``alert`` parameter).
    2. Redis deduplication — drop duplicate ``signal_id`` within 60 s.
    3. Volatility lock check — veto and log if macro lock is active.
    4. Desk state hydration — merge alert with live desk context.
    5. ClaudeCTO dispatch — obtain Consensus Score from the LLM.
    """
    redis: RedisStateManager = request.app.state.redis

    # ------------------------------------------------------------------
    # 1. Deduplication
    # ------------------------------------------------------------------
    is_new = await redis.check_duplicate_signal(alert.signal_id)
    if not is_new:
        logger.info("Duplicate signal dropped: %s", alert.signal_id)
        return ORJSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "duplicate",
                "signal_id": alert.signal_id,
                "message": "Signal already processed within dedup window.",
            },
        )

    # ------------------------------------------------------------------
    # 2. Volatility lock
    # ------------------------------------------------------------------
    if await redis.is_volatility_lock_active():
        logger.warning("Volatility lock ACTIVE — vetoing %s", alert.signal_id)

        async with _async_session() as db:
            db.add(
                VetoLog(
                    signal_id=alert.signal_id,
                    symbol=alert.symbol,
                    reason="volatility_lock",
                    raw_payload=alert.model_dump_json(),
                    vetoed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()

        return ORJSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "vetoed",
                "signal_id": alert.signal_id,
                "reason": "Global volatility lock is active.",
            },
        )

    # ------------------------------------------------------------------
    # 3. Desk state retrieval
    # ------------------------------------------------------------------
    desk_state = await redis.get_desk_state(alert.desk_id)
    desk_state.setdefault("desk_id", alert.desk_id)

    # ------------------------------------------------------------------
    # 4. Signal hydration
    # ------------------------------------------------------------------
    raw_payload = alert.model_dump()
    try:
        hydrated = await hydrate_signal(raw_payload, desk_state)
    except HydrationError as exc:
        logger.error("Hydration failed for %s: %s", alert.signal_id, exc)
        return ORJSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "hydration_error",
                "signal_id": alert.signal_id,
                "detail": str(exc),
            },
        )

    # ------------------------------------------------------------------
    # 5. ClaudeCTO dispatch
    # ------------------------------------------------------------------
    cto_result = await _dispatch_to_claude_cto(hydrated)

    logger.info(
        "Signal %s processed — consensus_score=%s",
        alert.signal_id,
        cto_result.get("consensus_score"),
    )

    return ORJSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "processed",
            "signal_id": alert.signal_id,
            "symbol": alert.symbol,
            "action": alert.action,
            "consensus": cto_result,
        },
    )
