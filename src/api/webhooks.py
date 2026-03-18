"""TradingView webhook router — main signal ingestion entry point.

Receives alerts from TradingView, deduplicates via Redis, checks the
global volatility lock, hydrates the payload with desk state, and
dispatches it to the ClaudeCTO agent for a Consensus Score evaluation.

All database operations flow through an ``AsyncSession`` injected via
FastAPI ``Depends``.  Redis is accessed strictly through
``request.app.state.redis``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncGenerator

import httpx
import orjson
import pandas as pd
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.services.signal_hydration import HydrationError, hydrate_signal
from src.services.telegram_broadcaster import broadcast_trade
from src.strategies.desk3_swing import Desk3SwingStrategy
from src.strategies.desk4_gold import Desk4GoldStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://user:password@localhost:5432/oniquant",
)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

logger = logging.getLogger("oniquant.webhooks")

# ---------------------------------------------------------------------------
# Async database engine + session factory
# ---------------------------------------------------------------------------

_engine = create_async_engine(POSTGRES_URL, pool_pre_ping=True, pool_size=5)
_async_session_factory = async_sessionmaker(
    _engine, class_=AsyncSession, expire_on_commit=False
)


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
# Dependency — AsyncSession via Depends
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a scoped ``AsyncSession`` and guarantee cleanup."""
    async with _async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


DBSession = Annotated[AsyncSession, Depends(get_db)]

# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------


class WebhookPayload(BaseModel):
    """Schema for incoming TradingView webhook payloads."""

    signal_id: str = Field(
        ..., min_length=1, description="Unique alert identifier"
    )
    symbol: str = Field(
        ..., min_length=1, description="Instrument (e.g. XAUUSD)"
    )
    action: str = Field(
        ..., pattern=r"^(BUY|SELL|CLOSE)$", description="Trade action"
    )
    desk_id: int = Field(default=1, ge=1, description="Target trading desk")
    price: float | None = Field(
        default=None, gt=0, description="Alert trigger price"
    )
    timeframe: str | None = Field(default=None, description="Chart timeframe")
    message: str | None = Field(
        default=None, description="Free-text context from alert"
    )


# ---------------------------------------------------------------------------
# ClaudeCTO dispatch
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


async def _get_http_client() -> httpx.AsyncClient:
    """Lazy-initialised shared ``httpx.AsyncClient``."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


async def _dispatch_to_claude_cto(hydrated_payload: bytes) -> dict[str, Any]:
    """Send the hydrated signal to the Claude API for Consensus Score.

    Parameters
    ----------
    hydrated_payload : bytes
        orjson-serialised payload produced by ``hydrate_signal()``.

    Returns
    -------
    dict
        Parsed model response with ``consensus_score`` and ``reasoning``
        keys, or an error dict on failure.
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        logger.warning("CLAUDE_API_KEY not set — skipping CTO dispatch")
        return {"consensus_score": None, "reasoning": "API key not configured"}

    # Deserialise the orjson bytes back into a dict so we can split out
    # the system prompt and build the Messages API payload.
    payload: dict = orjson.loads(hydrated_payload)
    system_prompt: str = payload.pop("system_prompt", "")

    client = await _get_http_client()
    response = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
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
        logger.error(
            "Claude API error %d: %s", response.status_code, response.text
        )
        return {
            "consensus_score": None,
            "reasoning": f"API error {response.status_code}",
        }

    body = response.json()
    text: str = body.get("content", [{}])[0].get("text", "")

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
async def tradingview_alert(
    payload: WebhookPayload,
    request: Request,
    db: DBSession,
):
    """Ingest a TradingView webhook alert through the full signal pipeline.

    Pipeline stages
    ---------------
    1.  Pydantic validation   (automatic via ``payload``).
    2.  Redis dedup           — drop duplicates, return 202.
    3.  Volatility lock       — veto + log to Postgres, return 202.
    4.  Desk state fetch      — ``get_desk_state(desk_id)``.
    4b. Strategy evaluation   — Desk 3 Lorentzian / Desk 4 Gold breakout.
    5.  Signal hydration      — ``hydrate_signal(dict, dict) -> bytes``.
    6.  ClaudeCTO dispatch    — Consensus Score evaluation.
    7.  Telegram broadcast    — fire-and-forget notification.
    """
    redis = request.app.state.redis

    # ------------------------------------------------------------------
    # Stage 2 — Deduplication
    # ------------------------------------------------------------------
    is_new: bool = await redis.check_duplicate_signal(payload.signal_id)
    if not is_new:
        logger.info("Duplicate signal dropped: %s", payload.signal_id)
        return ORJSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "duplicate",
                "signal_id": payload.signal_id,
                "message": "Signal already processed within dedup window.",
            },
        )

    # ------------------------------------------------------------------
    # Stage 3 — Volatility lock
    # ------------------------------------------------------------------
    if await redis.is_volatility_lock_active():
        logger.warning(
            "Volatility lock ACTIVE — vetoing %s", payload.signal_id
        )

        veto = VetoLog(
            signal_id=payload.signal_id,
            symbol=payload.symbol,
            reason="volatility_lock",
            raw_payload=payload.model_dump_json(),
            vetoed_at=datetime.now(timezone.utc),
        )
        db.add(veto)
        await db.commit()

        return ORJSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "vetoed",
                "signal_id": payload.signal_id,
                "reason": "Global volatility lock is active.",
            },
        )

    # ------------------------------------------------------------------
    # Stage 4 — Desk state retrieval
    # ------------------------------------------------------------------
    desk_state: dict = await redis.get_desk_state(payload.desk_id)
    desk_state.setdefault("desk_id", payload.desk_id)

    # ------------------------------------------------------------------
    # Stage 4b — Quantitative strategy evaluation
    # ------------------------------------------------------------------
    ohlcv = desk_state.get("ohlcv")
    df = pd.DataFrame(ohlcv) if ohlcv is not None else None

    if payload.desk_id == 3 and df is not None and not df.empty:
        strategy = Desk3SwingStrategy()
        result = strategy.evaluate(df)
        desk_state["is_valid_setup"] = result.is_valid_setup
        desk_state["lorentzian_score"] = result.lorentzian_score
        desk_state["swing_trend_direction"] = result.trend_direction
        desk_state["rsi"] = result.rsi
        desk_state["adx"] = result.adx
        desk_state["strategy"] = "desk3_lorentzian_swing"
        logger.info(
            "Desk 3 Lorentzian eval for %s: valid=%s score=%.4f",
            payload.signal_id,
            result.is_valid_setup,
            result.lorentzian_score,
        )

    elif payload.desk_id == 4 and df is not None and not df.empty:
        resistance = desk_state.get("resistance", 0.0)
        support = desk_state.get("support", 0.0)
        strategy = Desk4GoldStrategy()
        result = strategy.evaluate(df, resistance=resistance, support=support)
        desk_state["is_valid_breakout"] = result.is_valid_breakout
        desk_state["risk_reward_ratio"] = result.risk_reward_ratio
        desk_state["atr"] = result.atr
        desk_state["gold_trend_direction"] = result.trend_direction
        desk_state["strategy"] = "desk4_gold_breakout"
        logger.info(
            "Desk 4 Gold eval for %s: breakout=%s rr=%.4f",
            payload.signal_id,
            result.is_valid_breakout,
            result.risk_reward_ratio,
        )

    # ------------------------------------------------------------------
    # Stage 5 — Signal hydration
    # ------------------------------------------------------------------
    raw_dict: dict = payload.model_dump()
    try:
        hydrated: bytes = await hydrate_signal(raw_dict, desk_state)
    except HydrationError as exc:
        logger.error(
            "Hydration failed for %s: %s", payload.signal_id, exc
        )
        return ORJSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "hydration_error",
                "signal_id": payload.signal_id,
                "detail": str(exc),
            },
        )

    # ------------------------------------------------------------------
    # Stage 6 — ClaudeCTO dispatch
    # ------------------------------------------------------------------
    cto_result: dict = await _dispatch_to_claude_cto(hydrated)

    logger.info(
        "Signal %s processed — consensus_score=%s",
        payload.signal_id,
        cto_result.get("consensus_score"),
    )

    # ------------------------------------------------------------------
    # Stage 7 — Telegram broadcast (fire-and-forget)
    # ------------------------------------------------------------------
    await broadcast_trade(
        {
            "desk_id": payload.desk_id,
            "action": payload.action,
            "symbol": payload.symbol,
            "price": payload.price,
            "consensus_score": cto_result.get("consensus_score"),
        }
    )

    return ORJSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "processed",
            "signal_id": payload.signal_id,
            "symbol": payload.symbol,
            "action": payload.action,
            "consensus": cto_result,
        },
    )
