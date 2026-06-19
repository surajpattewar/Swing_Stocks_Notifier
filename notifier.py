"""
Alert senders. Telegram is the easiest free option (just a bot token + chat id).
WhatsApp uses Twilio's API — works instantly in Twilio's free sandbox for testing;
sending to arbitrary numbers in production requires WhatsApp Business approval.
"""
import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4000  # Telegram's hard limit is 4096 chars per message


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        logger.warning("Telegram not configured, skipping")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok = True
    # split into chunks if the message is long
    for i in range(0, len(text), TELEGRAM_MAX_LEN):
        chunk = text[i:i + TELEGRAM_MAX_LEN]
        try:
            resp = requests.post(url, data={"chat_id": chat_id, "text": chunk}, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            ok = False
    return ok


def send_whatsapp_twilio(account_sid: str, auth_token: str, from_whatsapp: str,
                          to_whatsapp: str, text: str) -> bool:
    if not all([account_sid, auth_token, from_whatsapp, to_whatsapp]):
        logger.warning("Twilio WhatsApp not configured, skipping")
        return False

    try:
        from twilio.rest import Client
    except ImportError:
        logger.error("twilio package not installed (pip install twilio)")
        return False

    client = Client(account_sid, auth_token)
    ok = True
    # WhatsApp messages via Twilio also have a length limit (~1600 chars is safe)
    chunk_size = 1500
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        try:
            client.messages.create(from_=from_whatsapp, to=to_whatsapp, body=chunk)
        except Exception as e:
            logger.error("WhatsApp send failed: %s", e)
            ok = False
    return ok


def format_alert_message(candidates: list, top_n: int) -> str:
    if not candidates:
        return "Swing Trade Screener\n\nNo qualifying setups found today."

    lines = [f"Swing Trade Screener — {len(candidates)} candidate(s) found on {datetime.today().date()} \nBy Suraj Pattewar"]
    for cand in candidates[:top_n]:
        lines.append(cand.to_line())
    lines.append(
        "\nAuto-generated results, not financial advice. "
    )
    return "\n\n".join(lines)