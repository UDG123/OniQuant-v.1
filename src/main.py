from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.services.redis_manager import RedisStateManager


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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
