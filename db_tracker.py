import os
import logging
import json
from datetime import datetime
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from config import config

logger = logging.getLogger(__name__)

# Keep a local SQLite connection fallback just in case get_connection() is called by un-migrated scripts
def get_connection():
    import sqlite3
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/screener_tracker.db")
    return conn, "sqlite"

def execute_query(conn, db_type, query, params=None):
    if params is None:
        params = []
    if db_type == "sqlite":
        query = query.replace("%s", "?")
    cursor = conn.cursor()
    cursor.execute(query, params)
    return cursor

# ------------------------------------------------------------------ #
# Google Sheets Connection & Client Helpers
# ------------------------------------------------------------------ #
def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # 1. Try credentials from environment variable JSON string
    if config.GOOGLE_SHEETS_CREDENTIALS_JSON:
        try:
            creds_dict = json.loads(config.GOOGLE_SHEETS_CREDENTIALS_JSON)
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            logger.error(f"Failed to load credentials from GOOGLE_SHEETS_CREDENTIALS_JSON: {e}")
            
    # 2. Try credentials from local JSON file path
    if os.path.exists(config.GOOGLE_SHEETS_CREDENTIALS_PATH):
        try:
            creds = Credentials.from_service_account_file(config.GOOGLE_SHEETS_CREDENTIALS_PATH, scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            logger.error(f"Failed to load credentials from file {config.GOOGLE_SHEETS_CREDENTIALS_PATH}: {e}")
            
    raise ValueError(
        "Google Sheets credentials not found. Please set GOOGLE_SHEETS_CREDENTIALS_JSON "
        "or GOOGLE_SHEETS_CREDENTIALS_PATH in your .env file."
    )

def get_spreadsheet(gc=None):
    if gc is None:
        gc = get_gspread_client()
    return gc.open(config.GOOGLE_SHEET_NAME)

# ------------------------------------------------------------------ #
# Google Sheets Initialization & CRUD Helpers
# ------------------------------------------------------------------ #
def init_db():
    """Initializes the spreadsheet and worksheets if they don't exist."""
    try:
        gc = get_gspread_client()
        try:
            sh = get_spreadsheet(gc)
        except gspread.SpreadsheetNotFound:
            logger.info(f"Spreadsheet '{config.GOOGLE_SHEET_NAME}' not found. Creating a new one...")
            sh = gc.create(config.GOOGLE_SHEET_NAME)
            logger.warning(
                f"Created new spreadsheet '{config.GOOGLE_SHEET_NAME}'. "
                "Remember to share it with your Google account / service account client email!"
            )
            
        existing_worksheets = [ws.title for ws in sh.worksheets()]
        
        # 1. daily_results
        if "daily_results" not in existing_worksheets:
            headers = ["date", "symbol", "setup_type", "score", "close", "stop_loss", "target", "reasons", "created_at"]
            ws = sh.add_worksheet("daily_results", rows=1000, cols=len(headers))
            ws.update(values=[headers], range_name='A1')
            logger.info("Created 'daily_results' worksheet.")
            
        # 2. open_positions
        if "open_positions" not in existing_worksheets:
            headers = ["symbol", "entry_date", "entry_price", "entry_sl", "entry_target", "current_sl", "current_target", "setup_type", "status", "current_price", "current_return", "close_date", "close_price", "updated_at"]
            ws = sh.add_worksheet("open_positions", rows=1000, cols=len(headers))
            ws.update(values=[headers], range_name='A1')
            logger.info("Created 'open_positions' worksheet.")
            
        # 3. position_progress
        if "position_progress" not in existing_worksheets:
            headers = ["symbol"] + [f"day{i}" for i in range(1, 16)] + ["updated_at"]
            ws = sh.add_worksheet("position_progress", rows=1000, cols=len(headers))
            ws.update(values=[headers], range_name='A1')
            logger.info("Created 'position_progress' worksheet.")
            
        logger.info("Database (Google Sheets) initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Google Sheets database: {e}")
        raise e

def _get_sheet_as_df(sheet_name: str) -> pd.DataFrame:
    try:
        sh = get_spreadsheet()
    except gspread.SpreadsheetNotFound:
        init_db()
        sh = get_spreadsheet()
        
    try:
        ws = sh.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        init_db()
        ws = sh.worksheet(sheet_name)
        
    values = ws.get_all_values()
    if len(values) > 1:
        headers = values[0]
        data = values[1:]
        df = pd.DataFrame(data, columns=headers)
    else:
        headers = values[0] if len(values) == 1 else []
        df = pd.DataFrame(columns=headers)
        
    return df

def _save_df_to_sheet(df: pd.DataFrame, sheet_name: str):
    sh = get_spreadsheet()
    ws = sh.worksheet(sheet_name)
    ws.clear()
    
    # Fill NaN/None with empty string, convert to string
    df_clean = df.fillna("").astype(str)
    headers = df_clean.columns.tolist()
    rows = df_clean.values.tolist()
    
    ws.update(values=[headers] + rows, range_name='A1')

# ------------------------------------------------------------------ #
# Business Data Readers
# ------------------------------------------------------------------ #
def get_daily_results() -> pd.DataFrame:
    return _get_sheet_as_df("daily_results")

def get_open_positions() -> pd.DataFrame:
    return _get_sheet_as_df("open_positions")

def get_position_progress() -> pd.DataFrame:
    return _get_sheet_as_df("position_progress")

# ------------------------------------------------------------------ #
# Business Data Writes / Logic
# ------------------------------------------------------------------ #
def save_daily_results(date_str, candidates):
    if not candidates:
        return
    try:
        df = get_daily_results()
        new_rows = []
        for cand in candidates:
            reasons_str = ", ".join(cand.reasons)
            
            # Deduplication: check if (date, symbol, setup_type) exists
            exists = False
            if not df.empty:
                match = df[
                    (df["date"] == date_str) & 
                    (df["symbol"] == cand.symbol) & 
                    (df["setup_type"] == cand.setup_type)
                ]
                if not match.empty:
                    exists = True
                    
            if exists:
                logger.info(f"Skipping duplicate daily result: {date_str}, {cand.symbol}, {cand.setup_type}")
                continue
                
            new_rows.append({
                "date": date_str,
                "symbol": cand.symbol,
                "setup_type": cand.setup_type,
                "score": int(cand.score),
                "close": float(cand.close),
                "stop_loss": float(cand.stop_loss),
                "target": float(cand.target),
                "reasons": reasons_str,
                "created_at": datetime.now().isoformat()
            })
            
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df = pd.concat([df, df_new], ignore_index=True)
            _save_df_to_sheet(df, "daily_results")
            logger.info(f"Saved {len(new_rows)} new candidates to daily_results sheet.")
    except Exception as e:
        logger.error(f"Error saving daily results: {e}")

def update_open_positions(date_str, candidates):
    if not candidates:
        return
    try:
        df_open = get_open_positions()
        df_prog = get_position_progress()
        
        if not df_open.empty:
            df_open["symbol"] = df_open["symbol"].astype(str)
        if not df_prog.empty:
            df_prog["symbol"] = df_prog["symbol"].astype(str)
            
        for cand in candidates:
            symbol = str(cand.symbol)
            
            # Find matching open position row index
            idx = -1
            if not df_open.empty:
                match = df_open[df_open["symbol"] == symbol]
                if not match.empty:
                    idx = match.index[0]
                    
            if idx != -1:
                status = df_open.at[idx, "status"]
                if status == "OPEN":
                    # Update active stock target & SL
                    df_open.at[idx, "current_sl"] = str(cand.stop_loss)
                    df_open.at[idx, "current_target"] = str(cand.target)
                    df_open.at[idx, "updated_at"] = datetime.now().isoformat()
                    logger.info(f"Updated SL/Target for active symbol {symbol} in Google Sheets.")
                else:
                    # Stock is CLOSED, reset it to OPEN for a new trade setup
                    df_open.at[idx, "entry_date"] = str(date_str)
                    df_open.at[idx, "entry_price"] = str(cand.close)
                    df_open.at[idx, "entry_sl"] = str(cand.stop_loss)
                    df_open.at[idx, "entry_target"] = str(cand.target)
                    df_open.at[idx, "current_sl"] = str(cand.stop_loss)
                    df_open.at[idx, "current_target"] = str(cand.target)
                    df_open.at[idx, "setup_type"] = str(cand.setup_type)
                    df_open.at[idx, "status"] = "OPEN"
                    df_open.at[idx, "current_price"] = str(cand.close)
                    df_open.at[idx, "current_return"] = "0.00%"
                    df_open.at[idx, "close_date"] = ""
                    df_open.at[idx, "close_price"] = ""
                    df_open.at[idx, "updated_at"] = datetime.now().isoformat()
                    logger.info(f"Re-opened closed symbol {symbol} for a new trade setup in Google Sheets.")
                    
                    # Clear position progress days
                    if not df_prog.empty:
                        p_match = df_prog[df_prog["symbol"] == symbol]
                        if not p_match.empty:
                            p_idx = p_match.index[0]
                            for i in range(1, 16):
                                df_prog.at[p_idx, f"day{i}"] = ""
                            df_prog.at[p_idx, "updated_at"] = datetime.now().isoformat()
            else:
                # Insert a new record
                new_pos = {
                    "symbol": symbol,
                    "entry_date": str(date_str),
                    "entry_price": str(cand.close),
                    "entry_sl": str(cand.stop_loss),
                    "entry_target": str(cand.target),
                    "current_sl": str(cand.stop_loss),
                    "current_target": str(cand.target),
                    "setup_type": str(cand.setup_type),
                    "status": "OPEN",
                    "current_price": str(cand.close),
                    "current_return": "0.00%",
                    "close_date": "",
                    "close_price": "",
                    "updated_at": datetime.now().isoformat()
                }
                df_open = pd.concat([df_open, pd.DataFrame([new_pos])], ignore_index=True)
                logger.info(f"Inserted new open position: {symbol} in Google Sheets.")
                
                # Check and initialize progress row
                p_exists = False
                if not df_prog.empty:
                    p_match = df_prog[df_prog["symbol"] == symbol]
                    p_exists = not p_match.empty
                    
                if not p_exists:
                    new_prog = {"symbol": symbol, "updated_at": datetime.now().isoformat()}
                    for i in range(1, 16):
                        new_prog[f"day{i}"] = ""
                    df_prog = pd.concat([df_prog, pd.DataFrame([new_prog])], ignore_index=True)
                    
        _save_df_to_sheet(df_open, "open_positions")
        _save_df_to_sheet(df_prog, "position_progress")
    except Exception as e:
        logger.error(f"Error updating open positions: {e}")

def track_daily_progress(date_str):
    try:
        df_open = get_open_positions()
        df_prog = get_position_progress()
        
        if df_open.empty:
            logger.info("No open positions to track progress.")
            return
            
        df_open["symbol"] = df_open["symbol"].astype(str)
        df_prog["symbol"] = df_prog["symbol"].astype(str)
        
        open_rows = df_open[df_open["status"] == "OPEN"]
        if open_rows.empty:
            logger.info("No active open positions to track progress.")
            return
            
        import yfinance as yf
        
        for idx, pos in open_rows.iterrows():
            symbol = pos["symbol"]
            entry_price = float(pos["entry_price"])
            current_sl = float(pos["current_sl"])
            current_target = float(pos["current_target"])
            
            today_close = None
            
            # 1. Check local DuckDB first for specific historical date (critical for backfilling)
            try:
                import duckdb
                db_path = config.DUCKDB_PATH
                if os.path.exists(db_path):
                    with duckdb.connect(db_path, read_only=True) as con_db:
                        row_db = con_db.execute(
                            "SELECT close FROM stock_prices WHERE symbol = ? AND CAST(timezone('Asia/Kolkata', date) AS DATE) = ?",
                            [symbol, date_str]
                        ).fetchone()
                        if row_db:
                            today_close = float(row_db[0])
            except Exception as ex:
                logger.debug(f"DuckDB lookup failed for {symbol} on {date_str}: {ex}")
                
            # 2. Try yfinance live data lookup
            if today_close is None:
                try:
                    ticker = yf.Ticker(symbol)
                    df_today = ticker.history(period="1d", auto_adjust=True)
                    if not df_today.empty:
                        today_close = float(df_today["Close"].iloc[-1])
                except Exception:
                    pass
                    
            # 3. DuckDB fallback to the latest available price if specific date and yfinance both failed
            if today_close is None:
                try:
                    import duckdb
                    db_path = config.DUCKDB_PATH
                    if os.path.exists(db_path):
                        with duckdb.connect(db_path, read_only=True) as con_db:
                            row_db = con_db.execute(
                                "SELECT close FROM stock_prices WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                                [symbol]
                            ).fetchone()
                            if row_db:
                                today_close = float(row_db[0])
                except Exception as ex:
                    logger.warning(f"DuckDB fallback failed for {symbol}: {ex}")
                    
            if today_close is None:
                logger.warning(f"Could not retrieve close price for {symbol} progress tracking.")
                continue
                
            # Find progress row
            p_idx = -1
            if not df_prog.empty:
                p_match = df_prog[df_prog["symbol"] == symbol]
                if not p_match.empty:
                    p_idx = p_match.index[0]
                    
            if p_idx == -1:
                new_prog = {"symbol": symbol, "updated_at": datetime.now().isoformat()}
                for i in range(1, 16):
                    new_prog[f"day{i}"] = ""
                df_prog = pd.concat([df_prog, pd.DataFrame([new_prog])], ignore_index=True)
                p_idx = df_prog.index[-1]
                
            # Determine target column (first empty dayX column)
            day_num = 1
            for i in range(1, 16):
                val = df_prog.at[p_idx, f"day{i}"]
                if val is None or str(val).strip() == "":
                    break
                day_num += 1
                
            if day_num > 15:
                # Timeout, close position
                df_open.at[idx, "status"] = "CLOSED"
                df_open.at[idx, "close_date"] = str(date_str)
                df_open.at[idx, "close_price"] = str(today_close)
                df_open.at[idx, "current_price"] = str(today_close)
                df_open.at[idx, "current_return"] = f"{((today_close - entry_price) / entry_price * 100):+.2f}%"
                df_open.at[idx, "updated_at"] = datetime.now().isoformat()
                logger.info(f"Closed position {symbol} (exceeded 15 tracking days) in Google Sheets.")
                continue
                
            # Calculate % change
            pct_change = (today_close - entry_price) / entry_price * 100
            pct_str = f"{pct_change:+.2f}%"
            
            # Update current live price and return
            df_open.at[idx, "current_price"] = str(today_close)
            df_open.at[idx, "current_return"] = pct_str
            
            outcome_str = pct_str
            closed = False
            close_price = today_close
            
            if today_close >= current_target:
                outcome_str = f"{pct_str} (Target Hit)"
                closed = True
            elif today_close <= current_sl:
                outcome_str = f"{pct_str} (SL Hit)"
                closed = True
            elif day_num == 15:
                outcome_str = f"{pct_str} (Timeout)"
                closed = True
                
            # Update position progress dayN column
            df_prog.at[p_idx, f"day{day_num}"] = str(outcome_str)
            df_prog.at[p_idx, "updated_at"] = datetime.now().isoformat()
            logger.info(f"Logged day {day_num} progress for {symbol}: {outcome_str} in Google Sheets.")
            
            if closed:
                df_open.at[idx, "status"] = "CLOSED"
                df_open.at[idx, "close_date"] = str(date_str)
                df_open.at[idx, "close_price"] = str(close_price)
                df_open.at[idx, "current_price"] = str(close_price)
                df_open.at[idx, "current_return"] = pct_str
                df_open.at[idx, "updated_at"] = datetime.now().isoformat()
                logger.info(f"Closed position {symbol} on day {day_num} in Google Sheets.")
                
        _save_df_to_sheet(df_open, "open_positions")
        _save_df_to_sheet(df_prog, "position_progress")
    except Exception as e:
        logger.error(f"Error tracking progress: {e}")
