"""
Central configuration. All secrets/settings come from environment variables
(loaded from a .env file locally, or from the host's secret store in the cloud).
Never hardcode tokens here.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # no-op if .env doesn't exist (e.g. in cloud where env vars are injected directly)


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


class Config:
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # Twilio WhatsApp
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")
    TWILIO_WHATSAPP_TO = os.getenv("TWILIO_WHATSAPP_TO", "")

    # Screener tuning
    MIN_SCORE = int(os.getenv("MIN_SCORE", "3"))               # 0-5, higher = stricter
    MAX_STOCKS_TO_SCAN = int(os.getenv("MAX_STOCKS_TO_SCAN", "500"))
    TOP_N_ALERTS = int(os.getenv("TOP_N_ALERTS", "15"))
    HISTORY_PERIOD = os.getenv("HISTORY_PERIOD", "9mo")
    HISTORY_INTERVAL = os.getenv("HISTORY_INTERVAL", "1d")

    SEND_TELEGRAM = _get_bool("SEND_TELEGRAM", False)
    SEND_WHATSAPP = _get_bool("SEND_WHATSAPP", False)

    # DuckDB persistence
    DUCKDB_PATH = os.getenv("DUCKDB_PATH", "data/duckdb/screener_data.duckdb")
    SAVE_TO_DUCKDB = _get_bool("SAVE_TO_DUCKDB", True)

    # Backtesting
    BACKTEST_MONTHS = int(os.getenv("BACKTEST_MONTHS", "4"))          # how far back to test
    BACKTEST_MAX_HOLDING_DAYS = int(os.getenv("BACKTEST_MAX_HOLDING_DAYS", "15"))  # trading days per signal
    BACKTEST_MIN_SCORE = int(os.getenv("BACKTEST_MIN_SCORE", str(MIN_SCORE)))
    BACKTEST_MAX_STOCKS = int(os.getenv("BACKTEST_MAX_STOCKS", "100"))
    BACKTEST_MAX_WORKERS = int(os.getenv("BACKTEST_MAX_WORKERS", "6"))


config = Config()
