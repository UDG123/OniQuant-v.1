"""Async Telegram broadcast service for trade notifications.

Sends MarkdownV2-formatted trade receipts to a configured Telegram chat
via the Bot API.  Failures are logged but never propagate — broadcasting
is fire-and-forget so it cannot block the trading pipeline.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("oniquant.telegram")

TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

_SEND_MESSAGE_URL = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    if TELEGRAM_BOT_TOKEN
    else ""
)

_ACTION_MAP: dict[str, str] = {
    "BUY": "LONG",
    "SELL": "SHORT",
    "CLOSE": "CLOSE",
}

_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=10.0)
    return _client


def _escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r"_*[]()~`>#+-=|{}.!"
    for ch in special:
        text = text.replace(ch, f"\\{ch}")
    return text


def _format_trade_message(receipt: dict) -> str:
    """Build a MarkdownV2 trade notification from a trade receipt dict.

    Expected keys: desk_id, action, symbol, price, consensus_score.
    """
    desk_id = receipt.get("desk_id", "—")
    raw_action = str(receipt.get("action", "UNKNOWN")).upper()
    action = _ACTION_MAP.get(raw_action, raw_action)
    symbol = str(receipt.get("symbol", "UNKNOWN"))
    price = receipt.get("price")
    consensus_score = receipt.get("consensus_score")

    price_str = f"{price:,.2f}" if isinstance(price, (int, float)) else "N/A"
    score_str = str(consensus_score) if consensus_score is not None else "N/A"

    return (
        f"*OniQuant Trade Executed*\n"
        f"\n"
        f"*Desk:*  {_escape_md(str(desk_id))}\n"
        f"*Action:*  {_escape_md(action)}\n"
        f"*Asset:*  {_escape_md(symbol)}\n"
        f"*Entry Price:*  {_escape_md(price_str)}\n"
        f"*Consensus Score:*  {_escape_md(score_str)}"
    )


async def broadcast_trade(trade_receipt: dict) -> None:
    """Send a formatted trade receipt to the configured Telegram chat.

    Parameters
    ----------
    trade_receipt : dict
        Must contain at minimum ``desk_id``, ``action``, ``symbol``.
        Optional: ``price``, ``consensus_score``.

    The call is fully fire-and-forget: errors are logged, never raised.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
            "skipping broadcast"
        )
        return

    text = _format_trade_message(trade_receipt)

    try:
        client = await _get_client()
        response = await client.post(
            _SEND_MESSAGE_URL,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": True,
            },
        )
        if response.status_code != 200:
            logger.error(
                "Telegram API %d: %s", response.status_code, response.text
            )
        else:
            logger.info(
                "Broadcast sent for %s %s",
                trade_receipt.get("symbol"),
                trade_receipt.get("action"),
            )
    except httpx.HTTPError as exc:
        logger.error("Telegram broadcast failed: %s", exc)
