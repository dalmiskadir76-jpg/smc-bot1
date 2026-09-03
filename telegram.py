"""Telegram notification sender."""
import logging
import os

import httpx

logger = logging.getLogger(__name__)


def _get_config():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
    return token, chat_id


def send_message(text: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    try:
        token, chat_id = _get_config()
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = httpx.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_notification": False,
        }, timeout=15)
        if resp.status_code == 200 and resp.json().get("ok"):
            logger.info("Telegram message sent successfully")
            return True
        logger.error("Telegram API error: %s", resp.text)
        return False
    except Exception:
        logger.exception("Failed to send Telegram message")
        return False


def send_startup_notification():
    text = (
        "\u2705 A+ SMC Monitor is LIVE\n"
        "\U0001f4ca Monitoring: XAUUSD | EURUSD | GBPUSD | USDJPY | AUDUSD | USDCAD\n"
        "\u23f1 Scan interval: Every 5 minutes\n"
        "\U0001f50d Waiting for A+ setups..."
    )
    return send_message(text)


def send_alert(pair_name: str, direction: str, reason: str, entry: float, sl: float, tp: float):
    side = "BUY" if direction == "bullish" else "SELL"
    text = (
        f"\U0001f6a8 A+ SETUP ALERT: {pair_name} \U0001f6a8\n\n"
        f"\U0001f4cd Type: {side}\n"
        f"\u23f1 Timeframe: M15 Sweep + M5 CHOCH/FVG\n"
        f"\U0001f4cb Reason: {reason}\n\n"
        f"\U0001f3af Key Levels:\n"
        f"  Entry (FVG Zone): {entry:.5f}\n"
        f"  Invalidation / SL: {sl:.5f}\n"
        f"  Target / TP (1:2 Min): {tp:.5f}\n\n"
        f"\u26a0\ufe0f Action Required: Open chart, capture screenshot, and verify before execution!"
    )
    return send_message(text)
