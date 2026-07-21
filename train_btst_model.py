import os
import sys
import json
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = config.DUCKDB_PATH
MODEL_DIR = os.path.join(workspace_dir, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "btst_stock_weights.json")

def main():
    logger.info("Starting stock-specific BTST parameter optimization...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    con = duckdb.connect(DB_PATH, read_only=True)
    symbols = get_stock_universe(max_stocks=200, no_of_stocks=200)
    
    latest_date_row = con.execute("SELECT max(date) FROM stock_prices").fetchone()
    end_date = pd.Timestamp(latest_date_row[0]).date()
    # Train: Start to 6 months ago; Test: Last 6 months
    test_start_date = end_date - timedelta(days=180)
    
    logger.info(f"Database End Date: {end_date}")
    logger.info(f"Walk-Forward Split Date: {test_start_date}")
    
    # Load Nifty index
    index_df = pd.DataFrame()
    try:
        index_raw = con.execute(
            """
            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                   close AS Close
            FROM stock_prices
            WHERE symbol = 'NSEI'
            ORDER BY date
            """
        ).fetchdf()
        if not index_raw.empty:
            index_raw["Date"] = pd.to_datetime(index_raw["Date"])
            index_df = index_raw.set_index("Date")
            index_df["nifty_sma50"] = ta.trend.sma_indicator(index_df["Close"], window=50)
            index_df = index_df.rename(columns={"Close": "nifty_close"})
    except Exception as e:
        logger.warning(f"Could not load Nifty: {e}")
        
    optimized_weights = {}
    
    # Param Grid to search per stock
    near_high_options = [0.002, 0.005, 0.010]  # 0.2%, 0.5%, 1.0% off high
    vol_ratio_options = [1.0, 1.2, 1.5, 2.0]   # Include volume contraction (1.0x) or mild spikes (1.2x)
    min_return_options = [1.0, 1.5, 2.0, 2.5]   # Support smaller daily candles
    rsi_ranges = [
        (55, 78),
        (60, 78),
        (65, 80),
        (50, 75)
    ]
    index_filters = ["sma50", "none"]
    
    for symbol in symbols:
        df = con.execute(
            """
            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                   open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
            FROM stock_prices
            WHERE symbol = ?
            ORDER BY date
            """,
            [symbol]
        ).fetchdf()
        
        if df.empty or len(df) < 100:
            continue
            
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        
        # Calculate indicators
        df["sma20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["vol_avg20"] = df["Volume"].shift(1).rolling(20).mean()
        df["rsi14"] = ta.momentum.rsi(df["Close"], window=14)
        
        df["next_open"] = df["Open"].shift(-1)
        df = df.dropna()
        
        # Derived attributes
        df["pct_today"] = (df["Close"] - df["Open"]) / df["Open"] * 100
        df["vol_ratio"] = df["Volume"] / df["vol_avg20"]
        df["off_high_pct"] = (df["High"] - df["Close"]) / df["High"] * 100
        df["uptrend"] = (df["Close"] > df["sma20"]) & (df["Close"] > df["sma50"])
        df["green_candle"] = df["Close"] > df["Open"]
        
        if not index_df.empty:
            df = df.join(index_df[["nifty_close", "nifty_sma50"]], how="left")
            df["index_ok"] = df["nifty_close"] > df["nifty_sma50"]
        else:
            df["index_ok"] = True
            
        best_cfg = None
        best_score = -1.0
        
        # Grid Search
        for nh in near_high_options:
            for vr in vol_ratio_options:
                for mr in min_return_options:
                    for r_min, r_max in rsi_ranges:
                        for idx_filt in index_filters:
                            
                            # Apply to full data
                            mask = (
                                df["green_candle"] &
                                df["uptrend"] &
                                (df["off_high_pct"] <= nh * 100) &
                                (df["vol_ratio"] >= vr) &
                                (df["pct_today"] >= mr) &
                                (df["rsi14"] >= r_min) &
                                (df["rsi14"] <= r_max)
                            )
                            if idx_filt == "sma50":
                                mask = mask & df["index_ok"]
                                
                            sub_df = df[mask]
                            total_trades = len(sub_df)
                            
                            if total_trades < 4: # Minimum 4 signals over the history
                                continue
                                
                            # Calculate Win Rate
                            ret_open = (sub_df["next_open"] - sub_df["Close"]) / sub_df["Close"] * 100
                            win_rate = (ret_open > 0).sum() / total_trades * 100
                            
                            if win_rate < 75.0: # Must be >= 75%
                                continue
                                
                            # Score: prioritize higher win rate, and higher trade count as a tie-breaker
                            score = win_rate * 100 + total_trades
                            if score > best_score:
                                best_score = score
                                best_cfg = {
                                    "near_high_pct": nh,
                                    "vol_ratio_limit": vr,
                                    "min_return": mr,
                                    "rsi_min": r_min,
                                    "rsi_max": r_max,
                                    "index_filter": idx_filt,
                                    "total_trades": int(total_trades),
                                    "win_rate": round(win_rate, 1)
                                }
                                    
        if best_cfg:
            optimized_weights[symbol] = best_cfg
            logger.info(f"Optimized custom weights for {symbol}: Win Rate {best_cfg['win_rate']}% on {best_cfg['total_trades']} trades")
            
    con.close()
    
    # Save parameters
    with open(MODEL_FILE, "w") as f:
        json.dump(optimized_weights, f, indent=4)
        
    logger.info(f"Successfully saved optimized custom BTST parameters for {len(optimized_weights)} stocks to: {MODEL_FILE}")

if __name__ == "__main__":
    main()
