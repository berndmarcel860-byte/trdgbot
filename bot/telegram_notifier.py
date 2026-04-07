"""Telegram Bot API notifier for trdgbot.

Provides a thin, dependency-free wrapper around the Telegram Bot HTTP API
using the ``requests`` library that is already in the project's dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from bot.logger import get_logger

logger = get_logger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    """Send and reply to Telegram messages via the Bot API.

    Args:
        token: Telegram Bot API token (``TELEGRAM_BOT_TOKEN``).
        channel_id: Target chat/channel ID (e.g. ``"@mychannel"`` or a numeric ID).
    """

    def __init__(self, token: str, channel_id: str) -> None:
        self._token = token
        self._channel_id = channel_id

    # ── Public API ─────────────────────────────────────────────────────────

    def send_message(self, text: str, parse_mode: str = "HTML") -> Optional[int]:
        """Send a message to the configured channel.

        Args:
            text: Message body (HTML or Markdown depending on *parse_mode*).
            parse_mode: ``"HTML"`` (default) or ``"MarkdownV2"``.

        Returns:
            The Telegram ``message_id`` of the sent message, or ``None`` on
            failure.
        """
        result = self._call(
            "sendMessage",
            {
                "chat_id": self._channel_id,
                "text": text,
                "parse_mode": parse_mode,
            },
        )
        if result:
            return int(result["message_id"])
        return None

    def reply_to(
        self,
        message_id: int,
        text: str,
        parse_mode: str = "HTML",
    ) -> Optional[int]:
        """Reply to an existing message in the channel.

        Args:
            message_id: The ``message_id`` to reply to.
            text: Reply body.
            parse_mode: ``"HTML"`` (default) or ``"MarkdownV2"``.

        Returns:
            The ``message_id`` of the reply, or ``None`` on failure.
        """
        result = self._call(
            "sendMessage",
            {
                "chat_id": self._channel_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_to_message_id": message_id,
            },
        )
        if result:
            return int(result["message_id"])
        return None

    # ── Internal ───────────────────────────────────────────────────────────

    def _call(self, method: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a Telegram Bot API call.

        Args:
            method: API method name (e.g. ``"sendMessage"``).
            payload: JSON body to POST.

        Returns:
            The ``result`` object from the response, or ``None`` on error.
        """
        url = _API_BASE.format(token=self._token, method=method)
        try:
            resp = requests.post(url, json=payload, timeout=10)
            data: Dict[str, Any] = resp.json()
        except Exception as exc:
            logger.error("Telegram HTTP request failed (%s): %s", method, exc)
            return None

        if not data.get("ok"):
            logger.warning(
                "Telegram API error [%s]: %s",
                method,
                data.get("description", "unknown error"),
            )
            return None

        return data.get("result")  # type: ignore[return-value]
