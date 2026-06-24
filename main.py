"""
One-shot run: build universe -> screen -> send alerts.
This is what gets executed daily (by cron / GitHub Actions / a scheduler).
Author : Suraj Pattewar
"""
import logging
import sys

from config import config
from stock_universe import get_stock_universe
from screener import run_screener as run_screener
from fundamental_screener import run_screener as run_fundamental_screener
from notifier import send_telegram, send_whatsapp_twilio, format_alert_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def trigger_technical_screener():
    # for no_of_stocks in [100, 200, 500]:
    for no_of_stocks in [100]:
        logger.info(f"Scanning Nifty {no_of_stocks} stocks...")

        candidates = run_screener(
            symbols=get_stock_universe(max_stocks=config.MAX_STOCKS_TO_SCAN, no_of_stocks=no_of_stocks),
            period=config.HISTORY_PERIOD,
            interval=config.HISTORY_INTERVAL,
            min_score=config.MIN_SCORE,
        )
        if candidates:
            break
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

def trigger_fundamental_screener():
    # for no_of_stocks in [100, 200, 500]:
    for no_of_stocks in [200]:
        logger.info(f"Scanning Nifty {no_of_stocks} stocks...")

        candidates = run_fundamental_screener(
            symbols=get_stock_universe(max_stocks=config.MAX_STOCKS_TO_SCAN, no_of_stocks=no_of_stocks),
            period=config.HISTORY_PERIOD,
            interval=config.HISTORY_INTERVAL,
            min_score=config.MIN_SCORE,
        )
        if candidates:
            break
    logger.info("Found %d stocks meeting min_score=%d", len(candidates), config.MIN_SCORE)

    message = format_alert_message(candidates, config.TOP_N_ALERTS, message="Fundamental screener")
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


def main():
    trigger_technical_screener()
    trigger_fundamental_screener()

if __name__ == "__main__":
    sys.exit(main())
