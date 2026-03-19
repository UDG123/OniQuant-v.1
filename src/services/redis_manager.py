from __future__ import annotations

import logging
import os
import time
from typing import Any

import orjson
import redis.asyncio as redis

logger = logging.getLogger("oniquant.redis_manager")

# Pending signals expire after 15 minutes (900 seconds).
_PENDING_SIGNAL_TTL_SECONDS = 900
_PENDING_QUEUE_KEY = "signals:pending"

# Pub/Sub channel for strategy parameter updates.
_STRATEGY_UPDATE_CHANNEL = "oniquant:strategy_updates"

# Emergency halt — global circuit breaker.
_EMERGENCY_HALT_CHANNEL = "oniquant:emergency_halt"
_VOLATILITY_LOCK_KEY = "global:volatility_lock"
_EMERGENCY_HALT_TTL_SECONDS = 86_400  # 24 hours

# ---------------------------------------------------------------------------
# CAS Lua script — Compare-And-Swap for parameter versioning
# ---------------------------------------------------------------------------
# KEYS[1] = active production key  (e.g. desk:1:params)
# ARGV[1] = expected current version (integer, "0" to skip check)
# ARGV[2] = new version (integer)
# ARGV[3] = serialised new parameters (bytes)
#
# Returns:
#   1  — swap succeeded
#   0  — version mismatch (stale write rejected)
# ---------------------------------------------------------------------------
_CAS_LUA_SCRIPT = """
local current_ver = redis.call('HGET', KEYS[1], '_version')
local expected    = ARGV[1]

-- If expected is "0" we skip the version check (initial write).
if expected ~= "0" then
    if current_ver == false then
        -- Key does not exist yet — reject unless caller expects version 0.
        return 0
    end
    if current_ver ~= expected then
        return 0
    end
end

-- Atomically overwrite the hash with new params + bumped version.
redis.call('DEL', KEYS[1])
redis.call('HSET', KEYS[1], '_version', ARGV[2], '_data', ARGV[3])
return 1
"""


class RedisStateManager:
    """Low-latency async Redis gateway for signal dedup, volatility locks, desk state, and atomic strategy hot-swap."""

    def __init__(self, url: str | None = None, max_connections: int = 50):
        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._pool: redis.ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._max_connections = max_connections
        self._cas_sha: str | None = None

    async def connect(self) -> None:
        self._pool = redis.ConnectionPool.from_url(
            self._url,
            max_connections=self._max_connections,
            decode_responses=False,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        # Pre-load the CAS Lua script into Redis so subsequent calls
        # use EVALSHA (single round-trip) instead of EVAL.
        self._cas_sha = await self._client.script_load(_CAS_LUA_SCRIPT)

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
        """Return True when the global volatility lock is engaged.

        The lock is active when the key exists with value ``b"TRUE"``
        (standard volatility lock) **or** ``b"HALT"`` (emergency halt).
        """
        value = await self._client.get(_VOLATILITY_LOCK_KEY)
        if value is None:
            return False
        return value in (b"TRUE", b"HALT", "TRUE", "HALT")

    async def is_emergency_halt_active(self) -> bool:
        """Return True only when the emergency halt state is engaged."""
        value = await self._client.get(_VOLATILITY_LOCK_KEY)
        if value is None:
            return False
        return value in (b"HALT", "HALT")

    async def activate_emergency_halt(self, reason: str) -> int:
        """Activate the global emergency halt.

        Sets ``global:volatility_lock`` to ``HALT`` with a 24-hour TTL
        and publishes the halt event on the ``oniquant:emergency_halt``
        Pub/Sub channel so all connected services can react immediately.

        Parameters
        ----------
        reason : str
            Human-readable explanation for the halt (persisted in the
            Pub/Sub message for audit logging).

        Returns
        -------
        int
            Number of Pub/Sub subscribers that received the halt message.
        """
        pipe = self._client.pipeline(transaction=True)
        pipe.set(
            _VOLATILITY_LOCK_KEY,
            "HALT",
            ex=_EMERGENCY_HALT_TTL_SECONDS,
        )
        halt_payload = orjson.dumps({
            "event": "EMERGENCY_HALT",
            "reason": reason,
            "ttl_seconds": _EMERGENCY_HALT_TTL_SECONDS,
            "issued_at": time.time(),
        })
        pipe.publish(_EMERGENCY_HALT_CHANNEL, halt_payload)
        results = await pipe.execute()

        subscriber_count: int = results[1]
        logger.warning(
            "EMERGENCY HALT activated — reason=%r ttl=%ds subscribers=%d",
            reason,
            _EMERGENCY_HALT_TTL_SECONDS,
            subscriber_count,
        )
        return subscriber_count

    async def deactivate_emergency_halt(self) -> bool:
        """Clear the emergency halt, restoring normal operation.

        Returns
        -------
        bool
            ``True`` if the key was present and deleted.
        """
        deleted: int = await self._client.delete(_VOLATILITY_LOCK_KEY)
        if deleted:
            logger.info("Emergency halt deactivated — normal operation resumed")
            # Notify subscribers that the halt has been lifted.
            await self._client.publish(
                _EMERGENCY_HALT_CHANNEL,
                orjson.dumps({
                    "event": "HALT_LIFTED",
                    "lifted_at": time.time(),
                }),
            )
        return deleted > 0

    async def subscribe_emergency_halt(self) -> redis.client.PubSub:
        """Return a Pub/Sub subscription for emergency halt events.

        Usage::

            pubsub = await redis_manager.subscribe_emergency_halt()
            async for message in pubsub.listen():
                if message["type"] == "message":
                    event = orjson.loads(message["data"])
                    if event["event"] == "EMERGENCY_HALT":
                        # cancel orders, close positions, etc.
        """
        pubsub = self._client.pubsub()
        await pubsub.subscribe(_EMERGENCY_HALT_CHANNEL)
        return pubsub

    async def get_desk_state(self, desk_id: int) -> dict:
        """Retrieve the full state hash for a trading desk."""
        raw: dict[bytes, bytes] = await self._client.hgetall(
            f"desk:{desk_id}:state"
        )
        return {k.decode(): _try_deserialize(v) for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Pending Signal Queue (ZSET — score = target_price)
    # ------------------------------------------------------------------

    async def add_pending_signal(
        self,
        payload: dict,
        target_price: float,
    ) -> bool:
        """Enqueue a signal that should trigger when *target_price* is hit.

        The payload is stored as a ZSET member with the target price as
        the score.  A ``expires_at`` epoch timestamp (now + 15 min) is
        injected into the payload so consumers can discard stale entries.

        Parameters
        ----------
        payload : dict
            Webhook payload (must be JSON-serialisable via orjson).
        target_price : float
            The price level that activates this signal.

        Returns
        -------
        bool
            ``True`` if the signal was newly added, ``False`` if it was
            already present (duplicate member in the ZSET).
        """
        enriched = {
            **payload,
            "target_price": target_price,
            "queued_at": time.time(),
            "expires_at": time.time() + _PENDING_SIGNAL_TTL_SECONDS,
        }
        member = orjson.dumps(enriched, option=orjson.OPT_SORT_KEYS)
        added: int = await self._client.zadd(
            _PENDING_QUEUE_KEY, {member: target_price}
        )
        return added > 0

    async def get_triggered_signals(
        self,
        current_price: float,
    ) -> list[dict]:
        """Fetch and atomically remove all signals whose target price has been crossed.

        Retrieves every ZSET member with ``score <= current_price``,
        removes them in a single pipeline round-trip, deserialises each
        member, and drops any that have exceeded their 15-minute TTL.

        Parameters
        ----------
        current_price : float
            The latest market price to compare against stored targets.

        Returns
        -------
        list[dict]
            Triggered (and still fresh) signal payloads.
        """
        # Atomic fetch + remove via a pipeline to prevent double-firing.
        pipe = self._client.pipeline(transaction=True)
        pipe.zrangebyscore(_PENDING_QUEUE_KEY, "-inf", current_price)
        pipe.zremrangebyscore(_PENDING_QUEUE_KEY, "-inf", current_price)
        results = await pipe.execute()

        raw_members: list[bytes] = results[0]
        now = time.time()
        triggered: list[dict] = []

        for member in raw_members:
            try:
                data: dict = orjson.loads(member)
            except (orjson.JSONDecodeError, ValueError):
                continue
            # Drop stale signals that outlived their 15-minute window.
            if data.get("expires_at", 0) < now:
                continue
            triggered.append(data)

        return triggered

    # ------------------------------------------------------------------
    # Atomic Strategy Hot-Swap
    # ------------------------------------------------------------------

    async def hot_swap_strategy_params(
        self,
        desk_id: int,
        new_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically replace strategy parameters for a desk.

        Execution flow:
            1. Write ``new_params`` to a shadow key
               ``tmp:desk:{desk_id}:params`` so production is untouched
               during the write.
            2. ``RENAME`` the shadow key onto the active production key
               ``desk:{desk_id}:params`` — this is an atomic O(1) pointer
               swap inside Redis, so readers never see a partial update.
            3. Publish a ``STRATEGY_UPDATE`` message on the
               ``oniquant:strategy_updates`` Pub/Sub channel so all
               connected workers can refresh their local memory cache
               instantly.

        Parameters
        ----------
        desk_id : int
            Target trading desk (1–5).
        new_params : dict[str, Any]
            Full parameter set to install.  Must be JSON-serialisable.

        Returns
        -------
        dict[str, Any]
            Receipt confirming the swap with ``desk_id``, ``version``,
            and ``params`` echoed back.
        """
        production_key = f"desk:{desk_id}:params"
        shadow_key = f"tmp:desk:{desk_id}:params"

        serialised = orjson.dumps(new_params, option=orjson.OPT_SORT_KEYS)

        # Fetch the current version so we can bump it.
        raw_version = await self._client.hget(production_key, "_version")
        current_version = int(raw_version) if raw_version else 0
        new_version = current_version + 1

        # Step 1 — Write to shadow key.
        await self._client.hset(
            shadow_key,
            mapping={
                b"_version": str(new_version).encode(),
                b"_data": serialised,
            },
        )

        # Step 2 — Atomic rename (shadow → production).
        await self._client.rename(shadow_key, production_key)

        # Step 3 — Publish update notification on Pub/Sub.
        notification = orjson.dumps({
            "event": "STRATEGY_UPDATE",
            "desk_id": desk_id,
            "version": new_version,
            "params": new_params,
        })
        subscriber_count = await self._client.publish(
            _STRATEGY_UPDATE_CHANNEL, notification,
        )

        logger.info(
            "Hot-swap complete: desk=%d version=%d→%d subscribers_notified=%d",
            desk_id,
            current_version,
            new_version,
            subscriber_count,
        )

        return {
            "desk_id": desk_id,
            "version": new_version,
            "previous_version": current_version,
            "subscribers_notified": subscriber_count,
            "params": new_params,
        }

    async def cas_strategy_params(
        self,
        desk_id: int,
        expected_version: int,
        new_params: dict[str, Any],
    ) -> bool:
        """Compare-And-Swap: update parameters only if the version matches.

        Uses a server-side Lua script so the read-compare-write is a
        single atomic operation — no WATCH/MULTI required.

        Parameters
        ----------
        desk_id : int
            Target trading desk.
        expected_version : int
            The version the caller believes is current.  Pass ``0`` to
            force-write regardless of current version (initial seed).
        new_params : dict[str, Any]
            New parameter set.

        Returns
        -------
        bool
            ``True`` if the swap succeeded, ``False`` if a concurrent
            writer already bumped the version (stale write rejected).
        """
        production_key = f"desk:{desk_id}:params"
        new_version = expected_version + 1 if expected_version > 0 else 1
        serialised = orjson.dumps(new_params, option=orjson.OPT_SORT_KEYS)

        result = await self._client.evalsha(
            self._cas_sha,
            1,
            production_key,
            str(expected_version),
            str(new_version),
            serialised,
        )

        success = int(result) == 1

        if success:
            # Broadcast on Pub/Sub after successful CAS.
            notification = orjson.dumps({
                "event": "STRATEGY_UPDATE",
                "desk_id": desk_id,
                "version": new_version,
                "params": new_params,
            })
            await self._client.publish(_STRATEGY_UPDATE_CHANNEL, notification)
            logger.info(
                "CAS swap succeeded: desk=%d version=%d→%d",
                desk_id, expected_version, new_version,
            )
        else:
            logger.warning(
                "CAS swap rejected: desk=%d expected_version=%d (stale)",
                desk_id, expected_version,
            )

        return success

    async def get_strategy_params(self, desk_id: int) -> tuple[int, dict[str, Any]]:
        """Read the current strategy parameters and version for a desk.

        Returns
        -------
        tuple[int, dict[str, Any]]
            ``(version, params)`` — version is ``0`` if no params are set.
        """
        production_key = f"desk:{desk_id}:params"
        raw: dict[bytes, bytes] = await self._client.hgetall(production_key)

        if not raw:
            return 0, {}

        version = int(raw.get(b"_version", b"0"))
        data_bytes = raw.get(b"_data", b"{}")

        try:
            params = orjson.loads(data_bytes)
        except (orjson.JSONDecodeError, ValueError):
            params = {}

        return version, params

    async def subscribe_strategy_updates(self) -> redis.client.PubSub:
        """Return a Pub/Sub subscription for strategy update notifications.

        Usage::

            pubsub = await redis_manager.subscribe_strategy_updates()
            async for message in pubsub.listen():
                if message["type"] == "message":
                    update = orjson.loads(message["data"])
                    # refresh local cache for update["desk_id"]
        """
        pubsub = self._client.pubsub()
        await pubsub.subscribe(_STRATEGY_UPDATE_CHANNEL)
        return pubsub

    async def purge_expired_signals(self) -> int:
        """Remove all pending signals whose 15-minute TTL has elapsed.

        Scans the full ZSET and drops expired members.  Intended to be
        called from a periodic background task so the queue stays lean.

        Returns
        -------
        int
            Number of expired members removed.
        """
        all_members: list[bytes] = await self._client.zrangebyscore(
            _PENDING_QUEUE_KEY, "-inf", "+inf"
        )
        now = time.time()
        expired: list[bytes] = []

        for member in all_members:
            try:
                data = orjson.loads(member)
            except (orjson.JSONDecodeError, ValueError):
                expired.append(member)
                continue
            if data.get("expires_at", 0) < now:
                expired.append(member)

        if expired:
            await self._client.zrem(_PENDING_QUEUE_KEY, *expired)

        return len(expired)


def _try_deserialize(value: bytes):
    """Attempt orjson decode; fall back to UTF-8 string."""
    try:
        return orjson.loads(value)
    except (orjson.JSONDecodeError, ValueError):
        return value.decode()
