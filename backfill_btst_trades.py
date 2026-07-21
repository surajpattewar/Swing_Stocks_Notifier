import os
import sys
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ta
import duckdb

# Add the workspace root to sys.path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from config import config
from stock_universe import get_stock_universe
from db_tracker import get_btst_trades, _save_df_to_sheet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = config.DUCKDB_PATH

def main():
    logger.info("Initializing BTST historical backfill (last 6 months)...")
    
    MODEL_FILE = os.path.join(workspace_dir, "models", "btst_stock_weights.json")
    custom_weights = {}
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "r") as f:
                custom_weights = json.load(f)
            logger.info(f"Loaded custom BTST parameters for {len(custom_weights)} stocks.")
        except Exception as e:
            logger.error(f"Failed to load custom weights: {e}")
            
    con = duckdb.connect(DB_PATH, read_only=True)
    
    # Get Nifty 200 symbols
    symbols = get_stock_universe(max_stocks=200, no_of_stocks=200)
    
    # Determine date range
    latest_date_row = con.execute("SELECT max(date) FROM stock_prices").fetchone()
    end_date = pd.Timestamp(latest_date_row[0]).date()
    start_date = end_date - timedelta(days=180) # 6 months
    
    logger.info(f"Backfill range: {start_date} to {end_date}")
    
    # Load Nifty index data
    index_df = pd.DataFrame()
    try:
        index_raw = con.execute(
            """
            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                   close AS Close
            FROM stock_prices
            WHERE symbol = 'NSEI' AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
            ORDER BY date
            """,
            [start_date - timedelta(days=60), end_date]
        ).fetchdf()
        if not index_raw.empty:
            index_raw["Date"] = pd.to_datetime(index_raw["Date"])
            index_df = index_raw.set_index("Date")
    except Exception as e:
        logger.warning(f"Could not load Nifty index data: {e}")
        
    if not index_df.empty:
        index_df["nifty_sma50"] = ta.trend.sma_indicator(index_df["Close"], window=50)
        index_df = index_df.rename(columns={"Close": "nifty_close"})
        
    backfill_records = []
    
    logger.info(f"Loading data for {len(symbols)} symbols...")
    for sym in symbols:
        df = con.execute(
            """
            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                   open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
            FROM stock_prices
            WHERE symbol = ? AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
            ORDER BY date
            """,
            [sym, start_date - timedelta(days=60), end_date]
        ).fetchdf()
        
        if df.empty or len(df) < 30:
            continue
            
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        
        # Calculate stock indicators
        df["sma20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["vol_avg20"] = df["Volume"].shift(1).rolling(20).mean()
        df["rsi14"] = ta.momentum.rsi(df["Close"], window=14)
        
        # Next day exits
        df["next_open"] = df["Open"].shift(-1)
        df["next_high"] = df["High"].shift(-1)
        df["next_low"] = df["Low"].shift(-1)
        df["next_close"] = df["Close"].shift(-1)
        
        df = df.dropna()
        df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
        
        if len(df) < 2:
            continue
            
        # Get custom parameters or defaults
        params = custom_weights.get(sym, {
            "near_high_pct": 0.002,
            "vol_ratio_limit": 1.5,
            "min_return": 2.0,
            "rsi_min": 60,
            "rsi_max": 78,
            "index_filter": "sma50"
        })
        nh_pct = params.get("near_high_pct", 0.002)
        vr_limit = params.get("vol_ratio_limit", 1.5)
        min_ret = params.get("min_return", 2.0)
        rsi_min = params.get("rsi_min", 60)
        rsi_max = params.get("rsi_max", 78)
        idx_filter = params.get("index_filter", "sma50")
            
        for dt, row in df.iterrows():
            pct_today = (row["Close"] - row["Open"]) / row["Open"] * 100
            near_high = row["Close"] >= (1.0 - nh_pct) * row["High"]
            vol_spike = row["Volume"] >= vr_limit * row["vol_avg20"]
            uptrend = row["Close"] > row["sma20"] and row["Close"] > row["sma50"]
            rsi_momentum = rsi_min <= row["rsi14"] <= rsi_max
            
            # Check broader index filter
            index_ok = True
            if idx_filter == "sma50" and not index_df.empty and dt in index_df.index:
                if dt in index_df["nifty_sma50"].index:
                    index_ok = index_df.loc[dt, "nifty_close"] > index_df.loc[dt, "nifty_sma50"]
                    
            if uptrend and near_high and vol_spike and row["Close"] > row["Open"] and pct_today >= min_ret and rsi_momentum and index_ok:
                entry_price = float(row["Close"])
                target_price = round(entry_price * 1.015, 2)
                sl_price = round(entry_price * 0.985, 2)
                
                n_open = float(row["next_open"])
                n_high = float(row["next_high"])
                n_low = float(row["next_low"])
                n_close = float(row["next_close"])
                
                # Exit A: Sell at Next Day's Open
                ret_open = (n_open - entry_price) / entry_price * 100
                
                # Exit B: Limit Target +1.5% or SL -1.5%, else Close
                if n_open >= target_price:
                    exit_price_b = n_open
                    ret_limit = (n_open - entry_price) / entry_price * 100
                    outcome = "WIN"
                elif n_open <= sl_price:
                    exit_price_b = n_open
                    ret_limit = (n_open - entry_price) / entry_price * 100
                    outcome = "LOSS"
                elif n_high >= target_price:
                    exit_price_b = target_price
                    ret_limit = 1.5
                    outcome = "WIN"
                elif n_low <= sl_price:
                    exit_price_b = sl_price
                    ret_limit = -1.5
                    outcome = "LOSS"
                else:
                    exit_price_b = n_close
                    ret_limit = (n_close - entry_price) / entry_price * 100
                    outcome = "WIN" if ret_limit > 0 else "LOSS"
                    
                backfill_records.append({
                    "date": str(dt.date()),
                    "symbol": sym,
                    "entry_price": round(entry_price, 2),
                    "target_1_5": round(target_price, 2),
                    "sl_1_5": round(sl_price, 2),
                    "rsi": round(float(row["rsi14"]), 1),
                    "vol_ratio": round(float(row["Volume"] / row["vol_avg20"]), 2),
                    "status": "CLOSED",
                    "next_open": round(n_open, 2),
                    "next_open_return": f"{ret_open:+.2f}%",
                    "exit_price_b": round(exit_price_b, 2),
                    "exit_return_b": f"{ret_limit:+.2f}%",
                    "outcome_b": outcome,
                    "created_at": datetime.now().isoformat()
                })
                
    con.close()
    
    if not backfill_records:
        logger.info("No matching BTST trades found in the past 6 months to backfill.")
        return
        
    df_new = pd.DataFrame(backfill_records)
    logger.info(f"Generated {len(df_new)} historical BTST trades from the past 6 months.")
    
    # Always save a local copy as a fallback
    os.makedirs("data", exist_ok=True)
    local_csv_path = "data/backfilled_btst_trades.csv"
    df_new.to_csv(local_csv_path, index=False)
    logger.info(f"Saved local copy of backfilled trades to: {local_csv_path}")
    
    # Save to Google Sheets
    try:
        df_existing = get_btst_trades()
        if not df_existing.empty:
            logger.info("Merging with existing trades in Google Sheet...")
            # Combine, and prioritize df_new if duplicates exist
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["date", "symbol"], keep="last")
        else:
            df_combined = df_new
            
        # Sort by date descending
        df_combined["date_parsed"] = pd.to_datetime(df_combined["date"])
        df_combined = df_combined.sort_values(by="date_parsed", ascending=False).drop(columns=["date_parsed"])
        
        _save_df_to_sheet(df_combined, "btst_trades")
        logger.info("Successfully uploaded backfilled BTST trades to 'btst_trades' worksheet!")
    except Exception as e:
        logger.error(f"Error uploading to Google Sheet: {e}")
        logger.info("You can manually upload the generated file 'data/backfilled_btst_trades.csv' to Google Sheets.")

if __name__ == "__main__":
    main()
