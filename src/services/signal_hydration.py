from __future__ import annotations

from datetime import datetime, timezone

import orjson

SYSTEM_PROMPT = (
    "You are a financial CTO evaluating live trading signals. "
    "Analyze the provided market data, desk state, and webhook payload. "
    "Output a JSON object with exactly two keys: "
    '"consensus_score" (integer 1-10, where 10 = strongest conviction) '
    'and "reasoning" (one concise sentence). '
    "Do not include any other commentary."
)


class HydrationError(Exception):
    """Raised when required fields are missing from input payloads."""


_REQUIRED_WEBHOOK_KEYS = ("signal_id", "symbol", "action")
_REQUIRED_DESK_KEYS = ("desk_id",)


def _validate_keys(payload: dict, required: tuple[str, ...], label: str) -> None:
    missing = [k for k in required if k not in payload]
    if missing:
        raise HydrationError(f"Missing keys in {label}: {missing}")


async def hydrate_signal(
    raw_webhook_payload: dict,
    redis_desk_state: dict,
) -> bytes:
    """Merge webhook + desk state and attach the CTO system prompt.

    Returns
    -------
    bytes
        orjson-serialized payload ready for the AI reasoning layer.

    Raises
    ------
    HydrationError
        If required keys are absent from either input dict.
    """
    _validate_keys(raw_webhook_payload, _REQUIRED_WEBHOOK_KEYS, "webhook_payload")
    _validate_keys(redis_desk_state, _REQUIRED_DESK_KEYS, "desk_state")

    hydrated = {
        "webhook": raw_webhook_payload,
        "desk_state": redis_desk_state,
        "system_prompt": SYSTEM_PROMPT,
        "hydrated_at": datetime.now(timezone.utc).isoformat(),
    }

    return orjson.dumps(hydrated, option=orjson.OPT_SORT_KEYS)
