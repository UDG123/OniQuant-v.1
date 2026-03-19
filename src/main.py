import asyncio
import logging
import os
from contextlib import asynccontextmanager

import optuna
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse
from optuna_dashboard import wsgi
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.webhooks import router as webhooks_router
from src.services.constitutional_manager import run_audit_loop
from src.services.redis_manager import RedisStateManager

logger = logging.getLogger("oniquant.main")

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
)
ASYNC_POSTGRES_URL = POSTGRES_URL.replace("postgresql://", "postgresql+asyncpg://")
SYNC_POSTGRES_URL = POSTGRES_URL.replace("+asyncpg", "")

# Async engine for health-check probes.
_health_engine = create_async_engine(
    ASYNC_POSTGRES_URL, pool_pre_ping=True, pool_size=2,
)
_health_session_factory = async_sessionmaker(
    _health_engine, class_=AsyncSession, expire_on_commit=False,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = RedisStateManager()
    await app.state.redis.connect()

    # Start the Constitutional Guardrail audit loop.
    app.state.constitutional_task = asyncio.create_task(run_audit_loop())

    yield

    app.state.constitutional_task.cancel()
    await app.state.redis.close()
    await _health_engine.dispose()


app = FastAPI(
    title="OniQuant",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(webhooks_router)

# ---------------------------------------------------------------------------
# Optuna Dashboard — mounted at /dashboard
# ---------------------------------------------------------------------------

_optuna_storage = optuna.storages.RDBStorage(
    url=SYNC_POSTGRES_URL,
    engine_kwargs={"pool_pre_ping": True, "pool_size": 3},
)
app.mount("/dashboard", WSGIMiddleware(wsgi(_optuna_storage)))


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/v1/health")
async def health_deep():
    """Deep health check — verifies both Redis and PostgreSQL connectivity.

    Returns 200 with per-service status when all checks pass, or 503
    with diagnostic detail when any dependency is unreachable.
    """
    checks: dict[str, str] = {}

    # --- Redis probe ---
    try:
        redis_mgr: RedisStateManager = app.state.redis
        await redis_mgr._client.ping()
        checks["redis"] = "healthy"
    except Exception as exc:
        checks["redis"] = f"unhealthy: {exc}"

    # --- PostgreSQL probe ---
    try:
        async with _health_session_factory() as db:
            await db.execute(text("SELECT 1"))
        checks["postgres"] = "healthy"
    except Exception as exc:
        checks["postgres"] = f"unhealthy: {exc}"

    all_healthy = all(v == "healthy" for v in checks.values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if all_healthy else "degraded",
            "checks": checks,
        },
    )


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
