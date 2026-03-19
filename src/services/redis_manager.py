"""Async Redis state manager — atomic hot-swap, signal dedup, volatility locks.

All strategy parameter mutations flow through server-side **Lua scripts**
that execute as single atomic operations inside the Redis event loop.
This eliminates TOCTOU race conditions that would arise from separate
``HGET`` → ``HSET`` round-trips in a multi-writer environment (Celery
workers, portfolio optimizer, champion-challenger orchestrator).

Two atomic primitives are provided:

* **CAS (Compare-And-Swap)** — ``_CAS_LUA`` reads the current
  ``_version``, compares it against the caller's expected version, and
  only writes if they match.  Stale writes are rejected with a ``0``
  return code, forcing the caller to re-read and retry.

* **Atomic Swap** — ``_ATOMIC_SWAP_LUA`` extends CAS with the
  Shadow-Key / ``RENAME`` pattern: the new config is first staged in
  a temporary hash (``tmp:desk:{id}:params``), then a single
  ``RENAME`` atomically replaces the production key.  Readers never
  observe a partial write because ``RENAME`` is an O(1) pointer swap.

After every successful mutation a ``CONFIG_RESET`` message is published
on the ``oniquant:config_reset`` Pub/Sub channel so all connected
workers can invalidate their local in-process caches immediately.
"""

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

# Pub/Sub channels.
_STRATEGY_UPDATE_CHANNEL = "oniquant:strategy_updates"
_CONFIG_RESET_CHANNEL = "oniquant:config_reset"

# Emergency halt — global circuit breaker.
_EMERGENCY_HALT_CHANNEL = "oniquant:emergency_halt"
_VOLATILITY_LOCK_KEY = "global:volatility_lock"
_EMERGENCY_HALT_TTL_SECONDS = 86_400  # 24 hours

# ---------------------------------------------------------------------------
# Lua script 1 — Compare-And-Swap (in-place, no shadow key)
# ---------------------------------------------------------------------------
# KEYS[1] = active production key  (e.g. desk:1:params)
# ARGV[1] = expected current version (integer, "0" to skip check)
# ARGV[2] = new version (integer)
# ARGV[3] = serialised new parameters (bytes)
#
# Returns:
#   {1, old_version}  — swap succeeded
#   {0, current_ver}  — version mismatch (stale write rejected)
# ---------------------------------------------------------------------------
_CAS_LUA = """
local prod_key    = KEYS[1]
local expected    = ARGV[1]
local new_ver     = ARGV[2]
local new_data    = ARGV[3]

local current_ver = redis.call('HGET', prod_key, '_version')
if current_ver == false then
    current_ver = '0'
end

-- If expected is "0" we skip the version check (initial seed write).
if expected ~= '0' and current_ver ~= expected then
    return {0, current_ver}
end

-- Atomically overwrite the hash with new params + bumped version.
redis.call('DEL', prod_key)
redis.call('HSET', prod_key, '_version', new_ver, '_data', new_data)
return {1, current_ver}
"""

# ---------------------------------------------------------------------------
# Lua script 2 — Atomic Swap via Shadow Key + RENAME + CAS
# ---------------------------------------------------------------------------
# KEYS[1] = production key       (e.g. desk:1:params)
# KEYS[2] = shadow key           (e.g. tmp:desk:1:params)
# ARGV[1] = expected version     (integer, "0" to force-write)
# ARGV[2] = new version          (integer)
# ARGV[3] = serialised new config (bytes)
#
# Flow executed atomically inside a single Lua invocation:
#   1. Read _version from the production key.
#   2. CAS: reject if current_ver != expected (unless expected == "0").
#   3. Stage the new config into the shadow key.
#   4. RENAME shadow → production (O(1) atomic pointer swap).
#
# Returns:
#   {1, old_version}  — swap succeeded, production key updated
#   {0, current_ver}  — version mismatch, production key untouched
# ---------------------------------------------------------------------------
_ATOMIC_SWAP_LUA = """
local prod_key    = KEYS[1]
local shadow_key  = KEYS[2]
local expected    = ARGV[1]
local new_ver     = ARGV[2]
local new_data    = ARGV[3]

-- Step 1: Read current version from production key.
local current_ver = redis.call('HGET', prod_key, '_version')
if current_ver == false then
    current_ver = '0'
end

-- Step 2: CAS guard — reject stale writes.
if expected ~= '0' and current_ver ~= expected then
    return {0, current_ver}
end

-- Step 3: Stage new config in the shadow key.
redis.call('DEL', shadow_key)
redis.call('HSET', shadow_key, '_version', new_ver, '_data', new_data)

-- Step 4: Atomic pointer swap — readers never see a partial state.
redis.call('RENAME', shadow_key, prod_key)

return {1, current_ver}
"""


class RedisStateManager:
    """Low-latency async Redis gateway for signal dedup, volatility locks,
    desk state, and atomic strategy hot-swap.

    All parameter mutations use server-side Lua scripts pre-loaded via
    ``SCRIPT LOAD`` at connect time, so subsequent calls use ``EVALSHA``
    (a single network round-trip) rather than transmitting the full
    script source on every invocation.
    """

    def __init__(self, url: str | None = None, max_connections: int = 50):
        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._pool: redis.ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._max_connections = max_connections
        self._cas_sha: str | None = None
        self._atomic_swap_sha: str | None = None

    async def connect(self) -> None:
        self._pool = redis.ConnectionPool.from_url(
            self._url,
            max_connections=self._max_connections,
            decode_responses=False,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        # Pre-load both Lua scripts into Redis so subsequent calls
        # use EVALSHA (single round-trip) instead of EVAL.
        self._cas_sha = await self._client.script_load(_CAS_LUA)
        self._atomic_swap_sha = await self._client.script_load(_ATOMIC_SWAP_LUA)

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
    # Atomic Strategy Hot-Swap (Shadow Key + RENAME + Lua CAS)
    # ------------------------------------------------------------------

    async def atomic_swap_params(
        self,
        desk_id: int,
        new_config: dict[str, Any],
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        """Atomically replace desk configuration via Shadow Key + CAS + RENAME.

        The entire read-compare-stage-swap sequence executes inside a
        **single Lua script invocation**, eliminating the TOCTOU race
        window that exists when version reads and shadow writes are
        separate Redis commands.

        Execution flow (all inside ``_ATOMIC_SWAP_LUA``):

            1. **CAS guard** — read ``_version`` from the production key
               (``desk:{id}:params``).  If *expected_version* is provided
               and doesn't match, the swap is rejected immediately
               (return code ``0``).
            2. **Shadow staging** — write the new config to
               ``tmp:desk:{id}:params`` so the production key is
               untouched until the swap completes.
            3. **Atomic RENAME** — ``RENAME tmp:desk:{id}:params
               desk:{id}:params`` is an O(1) pointer swap; readers
               never observe a partial write.

        After a successful swap, two Pub/Sub messages are broadcast:

        * ``STRATEGY_UPDATE`` on ``oniquant:strategy_updates`` — legacy
          notification consumed by existing workers.
        * ``CONFIG_RESET`` on ``oniquant:config_reset`` — explicit
          cache-invalidation signal that tells all connected workers to
          discard their in-process parameter caches for this desk.

        Parameters
        ----------
        desk_id : int
            Target trading desk (1–5).
        new_config : dict[str, Any]
            Full parameter / configuration set.  Must be JSON-serialisable.
        expected_version : int | None
            The version the caller believes is current.  When ``None``
            the current version is read first (unconditional swap).
            Pass an explicit version for strict optimistic concurrency.

        Returns
        -------
        dict[str, Any]
            Receipt with ``success``, ``desk_id``, ``version``,
            ``previous_version``, ``subscribers_notified``, and the
            echoed ``config``.

        Raises
        ------
        RuntimeError
            If the CAS check fails (version mismatch), indicating a
            concurrent writer modified the config between the caller's
            read and this swap attempt.
        """
        production_key = f"desk:{desk_id}:params"
        shadow_key = f"tmp:desk:{desk_id}:params"
        serialised = orjson.dumps(new_config, option=orjson.OPT_SORT_KEYS)

        # Determine expected version.
        if expected_version is None:
            raw_ver = await self._client.hget(production_key, "_version")
            expected_version = int(raw_ver) if raw_ver else 0

        new_version = expected_version + 1 if expected_version > 0 else 1

        # Execute the entire CAS + shadow + RENAME inside one Lua call.
        result = await self._client.evalsha(
            self._atomic_swap_sha,
            2,                          # number of KEYS
            production_key,
            shadow_key,
            str(expected_version),
            str(new_version),
            serialised,
        )

        success = int(result[0]) == 1
        old_version = int(result[1])

        if not success:
            logger.warning(
                "atomic_swap_params REJECTED: desk=%d expected=%d actual=%d "
                "(concurrent writer detected)",
                desk_id,
                expected_version,
                old_version,
            )
            raise RuntimeError(
                f"CAS version mismatch for desk {desk_id}: "
                f"expected {expected_version}, found {old_version}"
            )

        # --- Broadcast notifications in a single pipeline round-trip ------
        update_payload = orjson.dumps({
            "event": "STRATEGY_UPDATE",
            "desk_id": desk_id,
            "version": new_version,
            "previous_version": old_version,
            "config": new_config,
            "swapped_at": time.time(),
        })
        reset_payload = orjson.dumps({
            "event": "CONFIG_RESET",
            "desk_id": desk_id,
            "version": new_version,
            "reason": "atomic_swap",
            "invalidate_keys": [production_key],
            "issued_at": time.time(),
        })

        pipe = self._client.pipeline(transaction=False)
        pipe.publish(_STRATEGY_UPDATE_CHANNEL, update_payload)
        pipe.publish(_CONFIG_RESET_CHANNEL, reset_payload)
        pub_results = await pipe.execute()

        update_subs = int(pub_results[0])
        reset_subs = int(pub_results[1])

        logger.info(
            "atomic_swap_params OK: desk=%d version=%d→%d "
            "strategy_subs=%d config_reset_subs=%d",
            desk_id,
            old_version,
            new_version,
            update_subs,
            reset_subs,
        )

        return {
            "success": True,
            "desk_id": desk_id,
            "version": new_version,
            "previous_version": old_version,
            "subscribers_notified": update_subs + reset_subs,
            "config": new_config,
        }

    # ------------------------------------------------------------------
    # Legacy hot-swap (delegates to atomic_swap_params)
    # ------------------------------------------------------------------

    async def hot_swap_strategy_params(
        self,
        desk_id: int,
        new_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically replace strategy parameters for a desk.

        Thin wrapper around :meth:`atomic_swap_params` that preserves
        the original return-value shape for backward compatibility.

        Parameters
        ----------
        desk_id : int
            Target trading desk (1–5).
        new_params : dict[str, Any]
            Full parameter set to install.

        Returns
        -------
        dict[str, Any]
            Receipt confirming the swap.
        """
        receipt = await self.atomic_swap_params(
            desk_id=desk_id,
            new_config=new_params,
            expected_version=None,  # Unconditional swap.
        )
        return {
            "desk_id": receipt["desk_id"],
            "version": receipt["version"],
            "previous_version": receipt["previous_version"],
            "subscribers_notified": receipt["subscribers_notified"],
            "params": receipt["config"],
        }

    # ------------------------------------------------------------------
    # CAS — strict optimistic concurrency
    # ------------------------------------------------------------------

    async def cas_strategy_params(
        self,
        desk_id: int,
        expected_version: int,
        new_params: dict[str, Any],
    ) -> bool:
        """Compare-And-Swap: update parameters only if the version matches.

        Uses the ``_CAS_LUA`` server-side Lua script so the entire
        read-compare-write is a single atomic operation — no
        ``WATCH`` / ``MULTI`` required.

        On success, publishes both ``STRATEGY_UPDATE`` and
        ``CONFIG_RESET`` to notify all workers to invalidate caches.

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

        success = int(result[0]) == 1
        old_version = int(result[1])

        if success:
            # Broadcast both notifications in one pipeline.
            update_payload = orjson.dumps({
                "event": "STRATEGY_UPDATE",
                "desk_id": desk_id,
                "version": new_version,
                "params": new_params,
            })
            reset_payload = orjson.dumps({
                "event": "CONFIG_RESET",
                "desk_id": desk_id,
                "version": new_version,
                "reason": "cas_swap",
                "invalidate_keys": [production_key],
                "issued_at": time.time(),
            })
            pipe = self._client.pipeline(transaction=False)
            pipe.publish(_STRATEGY_UPDATE_CHANNEL, update_payload)
            pipe.publish(_CONFIG_RESET_CHANNEL, reset_payload)
            await pipe.execute()

            logger.info(
                "CAS swap succeeded: desk=%d version=%d→%d",
                desk_id, old_version, new_version,
            )
        else:
            logger.warning(
                "CAS swap rejected: desk=%d expected=%d actual=%d (stale)",
                desk_id, expected_version, old_version,
            )

        return success

    # ------------------------------------------------------------------
    # Parameter reads
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pub/Sub subscriptions
    # ------------------------------------------------------------------

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

    async def subscribe_config_reset(self) -> redis.client.PubSub:
        """Return a Pub/Sub subscription for ``CONFIG_RESET`` events.

        Workers should subscribe to this channel and discard any
        in-process parameter caches for the ``desk_id`` specified in
        the message payload.

        Usage::

            pubsub = await redis_manager.subscribe_config_reset()
            async for message in pubsub.listen():
                if message["type"] == "message":
                    event = orjson.loads(message["data"])
                    desk_id = event["desk_id"]
                    local_cache.pop(desk_id, None)
        """
        pubsub = self._client.pubsub()
        await pubsub.subscribe(_CONFIG_RESET_CHANNEL)
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
