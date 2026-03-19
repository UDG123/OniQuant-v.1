import os
from contextlib import asynccontextmanager

import optuna
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from optuna_dashboard import wsgi

from src.api.webhooks import router as webhooks_router
from src.services.redis_manager import RedisStateManager

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
)
SYNC_POSTGRES_URL = POSTGRES_URL.replace("+asyncpg", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = RedisStateManager()
    await app.state.redis.connect()
    yield
    await app.state.redis.close()


app = FastAPI(
    title="OniQuant",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(webhooks_router)

# ---------------------------------------------------------------------------
# Optuna Dashboard — mounted at /admin/optuna
# ---------------------------------------------------------------------------

_optuna_storage = optuna.storages.RDBStorage(
    url=SYNC_POSTGRES_URL,
    engine_kwargs={"pool_pre_ping": True, "pool_size": 3},
)
app.mount("/admin/optuna", WSGIMiddleware(wsgi(_optuna_storage)))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
