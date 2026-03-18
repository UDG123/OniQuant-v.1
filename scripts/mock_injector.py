"""Mock injector — simulate TradingView webhook alerts against the local pipeline.

Fires two identical requests to test both the happy path and the Redis
SETNX 60-second deduplication filter.

Usage:
    python -m scripts.mock_injector          # from project root
    python scripts/mock_injector.py          # direct execution
"""

from __future__ import annotations

import asyncio
import uuid

import httpx

ENDPOINT = "http://localhost:8000/api/v1/tradingview-alert"


def _build_payload(signal_id: str) -> dict:
    return {
        "signal_id": signal_id,
        "desk_id": 4,
        "symbol": "XAU/USD",
        "action": "BUY",
        "price": 2_345.67,
        "timeframe": "15m",
        "message": "Mock alert from injector script",
    }


async def main() -> None:
    signal_id = str(uuid.uuid4())
    payload = _build_payload(signal_id)

    print(f"Signal ID: {signal_id}\n")

    async with httpx.AsyncClient(timeout=15.0) as client:
        # -- First request: should pass through the full pipeline ----------
        print(">>> Request 1  (expect 200 processed)")
        r1 = await client.post(ENDPOINT, json=payload)
        print(f"    Status : {r1.status_code}")
        print(f"    Body   : {r1.json()}\n")

        # -- Second request: same signal_id → Redis dedup should reject ----
        print(">>> Request 2  (expect 202 duplicate)")
        r2 = await client.post(ENDPOINT, json=payload)
        print(f"    Status : {r2.status_code}")
        print(f"    Body   : {r2.json()}\n")

    # -- Summary -----------------------------------------------------------
    passed = r1.status_code == 200 and r2.status_code == 202
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] Deduplication test: first={r1.status_code}, second={r2.status_code}")


if __name__ == "__main__":
    asyncio.run(main())
