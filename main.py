"""
One-shot run: build universe -> screen -> send alerts.
This is what gets executed daily (by cron / GitHub Actions / a scheduler).
Author : Suraj Pattewar
"""
import logging
import sys

from config import config
from stock_universe import get_stock_universe
from screener import run_screener
from notifier import send_telegram, send_whatsapp_twilio, format_alert_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Building stock universe...")
    symbols = get_stock_universe(max_stocks=config.MAX_STOCKS_TO_SCAN)
    logger.info("Scanning %d stocks...", len(symbols))

    candidates = run_screener(
        symbols=symbols,
        period=config.HISTORY_PERIOD,
        interval=config.HISTORY_INTERVAL,
        min_score=config.MIN_SCORE,
    )
    logger.info("Found %d stocks meeting min_score=%d", len(candidates), config.MIN_SCORE)

    message = format_alert_message(candidates, config.TOP_N_ALERTS)
    print("\n" + message + "\n")

    if config.SEND_TELEGRAM:
        sent = send_telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, message)
        logger.info("Telegram sent: %s", sent)

    if config.SEND_WHATSAPP:
        sent = send_whatsapp_twilio(
            config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN,
            config.TWILIO_WHATSAPP_FROM, config.TWILIO_WHATSAPP_TO, message,
        )
        logger.info("WhatsApp sent: %s", sent)

    return 0


if __name__ == "__main__":
    sys.exit(main())
