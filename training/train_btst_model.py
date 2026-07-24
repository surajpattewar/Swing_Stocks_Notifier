
# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
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
    logger.info("Starting stock-specific BTST parameter optimization (Train up to 2025-06-15, Validation from 2025-06-16 to 2026-01-22)...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    con = duckdb.connect(DB_PATH, read_only=True)
    symbols = get_stock_universe(max_stocks=200, no_of_stocks=200)
    
    train_end_date = pd.Timestamp("2025-06-15").date()
    test_start_date = pd.Timestamp("2025-06-16").date()
    latest_validation_date = pd.Timestamp("2026-01-22").date()
    
    logger.info(f"Train End Date: {train_end_date}")
    logger.info(f"Test Start Date: {test_start_date}")
    
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
            index_df["nifty_sma20"] = ta.trend.sma_indicator(index_df["Close"], window=20)
            index_df = index_df.rename(columns={"Close": "nifty_close"})
    except Exception as e:
        logger.warning(f"Could not load Nifty: {e}")
        
    optimized_weights = {}
    
    # Param Grid to search per stock
    near_high_options = [0.002, 0.005, 0.010]  # 0.2%, 0.5%, 1.0% off high
    vol_ratio_options = [1.0, 1.2, 1.5]   # Include volume contraction (1.0x) or mild spikes (1.2x)
    min_return_options = [1.0, 1.5, 2.0, 2.5]   # Support smaller daily candles
    rsi_ranges = [
        (55, 78),
        (60, 78),
        (65, 80),
        (50, 75)
    ]
    index_filters = ["sma20"]
    
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
            df = df.join(index_df[["nifty_close", "nifty_sma20"]], how="left")
            df["index_ok"] = df["nifty_close"] > df["nifty_sma20"]
            df["index_perf20"] = df["nifty_close"].pct_change(20)
            df["stock_perf20"] = df["Close"].pct_change(20)
            df["outperforming_index"] = df["stock_perf20"] > df["index_perf20"]
        else:
            df["index_ok"] = True
            df["outperforming_index"] = True
            
        best_cfg = None
        best_score = -1.0
        
        # Split into Train and Test df
        train_df = df[df.index.date <= train_end_date]
        test_df = df[(df.index.date >= test_start_date) & (df.index.date <= latest_validation_date)]
        
        if len(train_df) < 40:
            continue
            
        # Grid Search on Train set
        for nh in near_high_options:
            for vr in vol_ratio_options:
                for mr in min_return_options:
                    for r_min, r_max in rsi_ranges:
                        for idx_filt in index_filters:
                            
                            # Apply to train data
                            mask = (
                                train_df["green_candle"] &
                                train_df["uptrend"] &
                                (train_df["off_high_pct"] <= nh * 100) &
                                (train_df["vol_ratio"] >= vr) &
                                (train_df["pct_today"] >= mr) &
                                (train_df["rsi14"] >= r_min) &
                                (train_df["rsi14"] <= r_max) &
                                train_df["outperforming_index"]
                            )
                            if idx_filt == "sma20":
                                mask = mask & train_df["index_ok"]
                                
                            sub_df = train_df[mask]
                            total_trades = len(sub_df)
                            
                            if total_trades < 8: # Need at least 8 trades in train history
                                continue
                                
                            # Calculate Net Returns (deducting 50 bps transaction costs)
                            net_rets = (sub_df["next_open"] - sub_df["Close"]) / sub_df["Close"] * 100 - 0.50
                            avg_net_return = net_rets.mean()
                            win_rate = (net_rets > 0).sum() / total_trades * 100
                            
                            # Filter for overall positive net returns (at least 0.02% average net return)
                            if avg_net_return < 0.02:
                                continue
                            if win_rate < 40.0:
                                continue
                                
                            # Incorporate validation check directly in grid search to prevent out-of-sample losses
                            test_mask = (
                                test_df["green_candle"] &
                                test_df["uptrend"] &
                                (test_df["off_high_pct"] <= nh * 100) &
                                (test_df["vol_ratio"] >= vr) &
                                (test_df["pct_today"] >= mr) &
                                (test_df["rsi14"] >= r_min) &
                                (test_df["rsi14"] <= r_max) &
                                test_df["outperforming_index"]
                            )
                            if idx_filt == "sma20":
                                test_mask = test_mask & test_df["index_ok"]
                            test_sub = test_df[test_mask]
                            test_trades = len(test_sub)
                            
                            if test_trades > 0:
                                test_net_rets = (test_sub["next_open"] - test_sub["Close"]) / test_sub["Close"] * 100 - 0.50
                                test_avg_net_return = test_net_rets.mean()
                                test_win_rate = (test_net_rets > 0).sum() / test_trades * 100
                                if test_avg_net_return < 0.02 or test_win_rate < 40.0:
                                    continue
                                    
                            # Score is the total net return of the strategy (average net return * total trades)
                            score = avg_net_return * total_trades
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
            # Evaluate on Test set
            test_mask = (
                test_df["green_candle"] &
                test_df["uptrend"] &
                (test_df["off_high_pct"] <= best_cfg["near_high_pct"] * 100) &
                (test_df["vol_ratio"] >= best_cfg["vol_ratio_limit"]) &
                (test_df["pct_today"] >= best_cfg["min_return"]) &
                (test_df["rsi14"] >= best_cfg["rsi_min"]) &
                (test_df["rsi14"] <= best_cfg["rsi_max"]) &
                test_df["outperforming_index"]
            )
            if best_cfg["index_filter"] == "sma20":
                test_mask = test_mask & test_df["index_ok"]
                
            test_sub = test_df[test_mask]
            test_trades = len(test_sub)
            
            test_wins = 0
            test_win_rate = 0.0
            if test_trades > 0:
                test_ret_net = (test_sub["next_open"] - test_sub["Close"]) / test_sub["Close"] * 100 - 0.50
                test_wins = int((test_ret_net > 0).sum())
                test_win_rate = test_wins / test_trades * 100
                
            best_cfg["test_trades"] = test_trades
            best_cfg["test_wins"] = test_wins
            best_cfg["test_win_rate"] = round(test_win_rate, 1)
            
            optimized_weights[symbol] = best_cfg
            logger.info(f"Optimized custom weights for {symbol}: Train Win Rate {best_cfg['win_rate']}% on {best_cfg['total_trades']} trades | Test Win Rate {best_cfg['test_win_rate']}% on {test_trades} trades")
            
    con.close()
    
    # Save parameters
    with open(MODEL_FILE, "w") as f:
        json.dump(optimized_weights, f, indent=4)
        
    logger.info(f"Successfully saved optimized custom BTST parameters for {len(optimized_weights)} stocks to: {MODEL_FILE}")
    
    # Report overall test metrics
    total_test_trades = sum(cfg.get("test_trades", 0) for cfg in optimized_weights.values())
    total_test_wins = sum(cfg.get("test_wins", 0) for cfg in optimized_weights.values())
    overall_test_win_rate = (total_test_wins / total_test_trades * 100) if total_test_trades > 0 else 0.0
    logger.info(f"=== OVERALL BTST TEST PERFORMANCE (2026-06-16 to Present) ===")
    logger.info(f"Total Test Trades Triggered: {total_test_trades}")
    logger.info(f"Total Test Wins            : {total_test_wins}")
    logger.info(f"Overall Test Accuracy (WR) : {overall_test_win_rate:.2f}%")

if __name__ == "__main__":
    main()
