import os
import pandas as pd
import logging
from db_tracker import get_spreadsheet

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def init_db_tuning():
    """Ensure the tuning_sheet worksheet exists in Google Sheets."""
    try:
        sh = get_spreadsheet()
        existing_worksheets = [ws.title for ws in sh.worksheets()]
        
        if "tuning_sheet" not in existing_worksheets:
            headers = ["date", "symbol", "setup_type", "score", "close", "stop_loss", "target", "reasons", "beta", "adx", "atr"]
            ws = sh.add_worksheet("tuning_sheet", rows=10000, cols=len(headers))
            ws.update(values=[headers], range_name='A1')
            logger.info("Created 'tuning_sheet' worksheet in Google Sheets.")
        else:
            logger.info("'tuning_sheet' worksheet already exists.")
    except Exception as e:
        logger.error(f"Error initializing tuning database: {e}")
        raise e

def save_tuning_signals(date_str, candidates_data):
    """Save daily candidates data list to the tuning_sheet worksheet."""
    if not candidates_data:
        return
    try:
        sh = get_spreadsheet()
        ws = sh.worksheet("tuning_sheet")
        
        # Load existing data
        records = ws.get_all_records()
        if records:
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame(columns=["date", "symbol", "setup_type", "score", "close", "stop_loss", "target", "reasons", "beta", "adx", "atr"])
        
        new_rows = []
        for cand in candidates_data:
            new_rows.append({
                "date": str(date_str),
                "symbol": str(cand["symbol"]),
                "setup_type": str(cand["setup_type"]),
                "score": int(cand["score"]),
                "close": float(cand["close"]),
                "stop_loss": float(cand["stop_loss"]),
                "target": float(cand["target"]),
                "reasons": str(cand["reasons"]),
                "beta": float(cand["beta"]),
                "adx": float(cand["adx"]),
                "atr": float(cand["atr"])
            })
            
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df = pd.concat([df, df_new], ignore_index=True)
            
            # Clean and write back
            df_clean = df.fillna("").astype(str)
            headers = df_clean.columns.tolist()
            rows = df_clean.values.tolist()
            
            ws.clear()
            ws.update(values=[headers] + rows, range_name='A1')
            logger.info(f"Successfully saved {len(new_rows)} signals to 'tuning_sheet'.")
    except Exception as e:
        logger.error(f"Error saving tuning signals: {e}")

def get_tuning_signals() -> pd.DataFrame:
    """Read tuning_sheet worksheet as a Pandas DataFrame."""
    try:
        sh = get_spreadsheet()
        ws = sh.worksheet("tuning_sheet")
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame(columns=["date", "symbol", "setup_type", "score", "close", "stop_loss", "target", "reasons", "beta", "adx", "atr"])
        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"Error reading tuning signals: {e}")
        return pd.DataFrame()

def clear_tuning_sheet():
    """Clear all records in tuning_sheet while keeping the headers."""
    try:
        sh = get_spreadsheet()
        ws = sh.worksheet("tuning_sheet")
        headers = ws.row_values(1)
        ws.clear()
        if headers:
            ws.update(values=[headers], range_name='A1')
        logger.info("Cleared 'tuning_sheet' worksheet (kept headers).")
    except Exception as e:
        logger.error(f"Error clearing 'tuning_sheet': {e}")
