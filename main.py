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
    from db_tracker import init_db, save_daily_results, update_open_positions, track_daily_progress, save_btst_trades, update_btst_trades
    import datetime

    # Initialize tracker database
    init_db()


    logger.info(f"Scanning Nifty 200 stocks...")

    candidates = run_screener(
        symbols=get_stock_universe(max_stocks=config.MAX_STOCKS_TO_SCAN, no_of_stocks=config.NIFTY_N_STOCKS),
        period=config.HISTORY_PERIOD,
        interval=config.HISTORY_INTERVAL,
        min_score=config.MIN_SCORE,
    )

    logger.info("Found %d stocks meeting min_score=%d", len(candidates), config.MIN_SCORE)

    # Run BTST Screener
    btst_candidates = []
    try:
        from btst_screener import run_btst_screener
        btst_candidates = run_btst_screener(send_alerts=False)
    except Exception as e:
        logger.error("Failed to run BTST screener: %s", e)

    date_str = str(datetime.date.today())
    
    # Sort by score and adx descending and cap to TOP_N_ALERTS (e.g. 5-7)
    candidates.sort(key=lambda c: (c.score, c.adx), reverse=True)
    candidates = candidates[:config.TOP_N_ALERTS]

    # Save today's candidates, open positions, and track progress daywise
    save_daily_results(date_str, candidates)
    update_open_positions(date_str, candidates)
    track_daily_progress(date_str)

    # Save today's BTST trades, and update previous day's open BTST trades
    try:
        update_btst_trades(date_str)
        save_btst_trades(date_str, btst_candidates)
    except Exception as e:
        logger.error("Failed to save/update BTST trades: %s", e)

    message = format_alert_message(candidates, config.TOP_N_ALERTS, btst_candidates=btst_candidates)
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
    # trigger_fundamental_screener()

if __name__ == "__main__":
    sys.exit(main())
