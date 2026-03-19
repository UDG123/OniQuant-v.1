"""TradingView webhook router — main signal ingestion entry point.

Receives alerts from TradingView, deduplicates via Redis, checks the
global volatility lock, hydrates the payload with desk state, and
dispatches it to the ClaudeCTO agent for a Consensus Score evaluation.

When the Consensus Score meets the execution threshold (>= 7.0) the
router either executes immediately via a dynamic limit order or defers
to the Redis pending-signal queue when a ``target_price`` has not yet
been reached.

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

from src.services.execution import (
    ExecutionError,
    OrderBroker,
    execute_dynamic_limit_order,
)
from src.services.signal_hydration import HydrationError, hydrate_signal
from src.services.telegram_broadcaster import broadcast_trade
from src.strategies.desk1_scalping import Desk1ScalpingStrategy
from src.strategies.desk2_fx import Desk2FXStrategy
from src.strategies.desk3_swing import Desk3SwingStrategy
from src.strategies.desk4_gold import Desk4GoldStrategy
from src.strategies.desk5_crypto import Desk5CryptoStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://user:password@localhost:5432/oniquant",
)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
CONSENSUS_THRESHOLD: float = float(os.getenv("CONSENSUS_THRESHOLD", "7.0"))

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
    target_price: float | None = Field(
        default=None, gt=0, description="Deferred execution target price"
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
    4b. Strategy evaluation   — All 5 desks: OFI / Kalman / Lorentzian / Gold / CVD.
    5.  Signal hydration      — ``hydrate_signal(dict, dict) -> bytes``.
    6.  ClaudeCTO dispatch    — Consensus Score evaluation.
    7.  Execution gate        — score >= 7.0: execute or defer to pending queue.
    8.  Telegram broadcast    — fire-and-forget notification.
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
    # Stage 4b — Quantitative strategy evaluation (all 5 desks)
    # ------------------------------------------------------------------
    ohlcv = desk_state.get("ohlcv")
    df = pd.DataFrame(ohlcv) if ohlcv is not None else None
    account_equity = desk_state.get("account_equity", 100_000.0)

    if df is not None and not df.empty:
        if payload.desk_id == 1:
            strategy = Desk1ScalpingStrategy()
            result = strategy.evaluate(df, account_equity=account_equity)
            desk_state["ofi_score"] = result.ofi_score
            desk_state["ofi_raw"] = result.ofi_raw
            desk_state["price_delta_pct"] = result.price_delta_pct
            desk_state["dynamic_stop"] = result.dynamic_stop
            desk_state["is_valid_setup"] = result.is_valid_setup
            desk_state["strategy_signal"] = result.signal
            desk_state["strategy"] = "desk1_ofi_scalp"
            logger.info(
                "Desk 1 OFI eval for %s: signal=%s ofi=%.4f valid=%s",
                payload.signal_id,
                result.signal,
                result.ofi_score,
                result.is_valid_setup,
            )

        elif payload.desk_id == 2:
            strategy = Desk2FXStrategy()
            result = strategy.evaluate(df, account_equity=account_equity)
            desk_state["kalman_z_score"] = result.kalman_z_score
            desk_state["kalman_price"] = result.kalman_price
            desk_state["rsi"] = result.rsi
            desk_state["atr"] = result.atr
            desk_state["dynamic_stop"] = result.dynamic_stop
            desk_state["is_valid_setup"] = result.is_valid_setup
            desk_state["strategy_signal"] = result.signal
            desk_state["strategy"] = "desk2_kalman_fx"
            logger.info(
                "Desk 2 Kalman eval for %s: signal=%s z=%.4f valid=%s",
                payload.signal_id,
                result.signal,
                result.kalman_z_score,
                result.is_valid_setup,
            )

        elif payload.desk_id == 3:
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

        elif payload.desk_id == 4:
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

        elif payload.desk_id == 5:
            strategy = Desk5CryptoStrategy()
            result = strategy.evaluate(df, account_equity=account_equity)
            desk_state["cvd_zscore"] = result.cvd_zscore
            desk_state["cvd_momentum"] = result.cvd_zscore
            desk_state["volatility_regime"] = result.volatility_regime
            desk_state["rsi"] = result.rsi
            desk_state["atr"] = result.atr
            desk_state["dynamic_stop"] = result.dynamic_stop
            desk_state["is_valid_setup"] = result.is_valid_setup
            desk_state["strategy_signal"] = result.signal
            desk_state["strategy"] = "desk5_cvd_crypto"
            logger.info(
                "Desk 5 CVD eval for %s: signal=%s z=%.4f regime=%s",
                payload.signal_id,
                result.signal,
                result.cvd_zscore,
                result.volatility_regime,
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
    # Stage 7 — Execution gate (consensus >= 7.0)
    # ------------------------------------------------------------------
    consensus_score = cto_result.get("consensus_score")
    execution_receipt: dict[str, Any] | None = None

    if (
        consensus_score is not None
        and isinstance(consensus_score, (int, float))
        and consensus_score >= CONSENSUS_THRESHOLD
    ):
        signal_dict: dict = payload.model_dump()
        signal_dict.update({
            "desk_id": payload.desk_id,
            "consensus_score": consensus_score,
        })

        # Check whether execution should be deferred to target_price.
        current_price = payload.price
        target_price = payload.target_price

        if (
            target_price is not None
            and current_price is not None
            and current_price < target_price
        ):
            # --- Defer: target not yet reached → enqueue in Redis ZSET ---
            enqueued: bool = await redis.add_pending_signal(
                payload=signal_dict,
                target_price=target_price,
            )
            logger.info(
                "Signal %s deferred to pending queue (target=%.4f, "
                "current=%.4f, enqueued=%s)",
                payload.signal_id,
                target_price,
                current_price,
                enqueued,
            )
            execution_receipt = {
                "status": "deferred",
                "target_price": target_price,
                "current_price": current_price,
                "enqueued": enqueued,
            }
        else:
            # --- Execute now: target met or no target specified -----------
            broker: OrderBroker = request.app.state.broker
            order_book: dict = await broker.get_order_book(payload.symbol)

            try:
                execution_receipt = await execute_dynamic_limit_order(
                    signal=signal_dict,
                    current_order_book=order_book,
                    broker=broker,
                    db=db,
                )
                logger.info(
                    "Signal %s executed → %s @ %s (%d chase cycles)",
                    payload.signal_id,
                    execution_receipt.get("state"),
                    execution_receipt.get("filled_price"),
                    execution_receipt.get("chase_cycles", 0),
                )
            except ExecutionError as exc:
                logger.error(
                    "Execution failed for %s: %s", payload.signal_id, exc
                )
                execution_receipt = {
                    "status": "execution_error",
                    "detail": str(exc),
                }
    else:
        logger.info(
            "Signal %s below consensus threshold (score=%s, required>=%.1f) "
            "— skipping execution",
            payload.signal_id,
            consensus_score,
            CONSENSUS_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # Stage 8 — Telegram broadcast (fire-and-forget)
    # ------------------------------------------------------------------
    await broadcast_trade(
        {
            "desk_id": payload.desk_id,
            "action": payload.action,
            "symbol": payload.symbol,
            "price": payload.price,
            "consensus_score": consensus_score,
        }
    )

    response_content: dict[str, Any] = {
        "status": "processed",
        "signal_id": payload.signal_id,
        "symbol": payload.symbol,
        "action": payload.action,
        "consensus": cto_result,
    }
    if execution_receipt is not None:
        response_content["execution"] = execution_receipt

    return ORJSONResponse(
        status_code=status.HTTP_200_OK,
        content=response_content,
    )
