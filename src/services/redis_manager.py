from __future__ import annotations

import os

import orjson
import redis.asyncio as redis


class RedisStateManager:
    """Low-latency async Redis gateway for signal dedup, volatility locks, and desk state."""

    def __init__(self, url: str | None = None, max_connections: int = 50):
        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._pool: redis.ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._max_connections = max_connections

    async def connect(self) -> None:
        self._pool = redis.ConnectionPool.from_url(
            self._url,
            max_connections=self._max_connections,
            decode_responses=False,
        )
        self._client = redis.Redis(connection_pool=self._pool)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Hot-path methods
    # ------------------------------------------------------------------

    async def check_duplicate_signal(self, signal_id: str) -> bool:
        """Return True if this signal is new (not a duplicate).

        Uses SET NX with a 60-second TTL for deduplication.
        Returns False when the key already exists (duplicate).
        """
        result = await self._client.set(
            f"signal:dedup:{signal_id}", b"1", nx=True, ex=60
        )
        return result is not None

    async def is_volatility_lock_active(self) -> bool:
        """Return True when the global volatility lock is engaged."""
        value = await self._client.get("global:volatility_lock")
        return value is not None

    async def get_desk_state(self, desk_id: int) -> dict:
        """Retrieve the full state hash for a trading desk."""
        raw: dict[bytes, bytes] = await self._client.hgetall(
            f"desk:{desk_id}:state"
        )
        return {k.decode(): _try_deserialize(v) for k, v in raw.items()}


def _try_deserialize(value: bytes):
    """Attempt orjson decode; fall back to UTF-8 string."""
    try:
        return orjson.loads(value)
    except (orjson.JSONDecodeError, ValueError):
        return value.decode()
