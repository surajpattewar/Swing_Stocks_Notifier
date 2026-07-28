#!/usr/bin/env python3

# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import json
import argparse
import logging
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import ta
import duckdb

from config import config
from stock_universe import get_stock_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_btst_model")

DB_PATH = config.DUCKDB_PATH
workspace_dir = _parent_dir
MODEL_DIR = os.path.join(workspace_dir, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "btst_stock_weights.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize stock-specific BTST parameters for high win rate and positive returns.")
    parser.add_argument("--min-sample-trades", type=int, default=int(os.getenv("BTST_MIN_SAMPLE_TRADES", "5")),
                        help="Minimum training trades per stock (default: 5)")
    parser.add_argument("--min-train-win-rate", type=float, default=float(os.getenv("BTST_MIN_TRAIN_WIN_RATE", "80.0")),
                        help="Minimum required training win rate % (default: 80.0%)")
    parser.add_argument("--cost-pct", type=float, default=0.25,
                        help="Slippage and transaction cost percent per trade (default: 0.25%, i.e., 0.50% round trip)")
    return parser.parse_args()

def compute_exit_type_b_returns(sub_df, round_trip_cost=0.50, target_pct=2.0, sl_pct=1.2):
    if sub_df.empty:
        return pd.Series(dtype='float64')
    entry = sub_df["Close"]
    n_open = sub_df["next_open"]
    n_high = sub_df["next_high"]
    n_low = sub_df["next_low"]
    n_close = sub_df["next_close"]
    
    target_price = entry * (1.0 + target_pct / 100.0)
    sl_price = entry * (1.0 - sl_pct / 100.0)
    
    cond_gap_up = n_open >= target_price
    cond_gap_down = (~cond_gap_up) & (n_open <= sl_price)
    cond_target_hit = (~cond_gap_up) & (~cond_gap_down) & (n_high >= target_price)
    cond_stop_hit = (~cond_gap_up) & (~cond_gap_down) & (~cond_target_hit) & (n_low <= sl_price)
    
    ret_limit = pd.Series(0.0, index=sub_df.index)
    ret_limit[cond_gap_up] = (n_open - entry) / entry * 100
    ret_limit[cond_gap_down] = (n_open - entry) / entry * 100
    ret_limit[cond_target_hit] = target_pct
    ret_limit[cond_stop_hit] = -sl_pct
    
    cond_else = (~cond_gap_up) & (~cond_gap_down) & (~cond_target_hit) & (~cond_stop_hit)
    ret_limit[cond_else] = (n_close - entry) / entry * 100
    
    return ret_limit - round_trip_cost

def get_signals_for_stock(df, nh, vr, mr, r_min, r_max, idx_filter, delivery_margin):
    if df.empty:
        return df.head(0)
        
    mask = (
        df["green_candle"] &
        df["uptrend"] &
        (df["off_high_pct"] <= nh * 100) &
        (df["vol_ratio"] >= vr) &
        (df["pct_today"] >= mr) &
        (df["rsi14"] >= r_min) &
        (df["rsi14"] <= r_max) &
        df["outperforming_index"] &
        df["liquid"] &
        (~df["circuit_locked"])
    )
    
    if idx_filter == "sma20":
        mask = mask & df["index_sma20_ok"]
    elif idx_filter == "sma50":
        mask = mask & df["index_sma50_ok"]
    elif idx_filter == "nifty_bull":
        mask = mask & df["index_sma20_ok"] & (df["nifty_ret5"] > 0)
        
    if "DeliveryPct" in df.columns and "deliv_avg20" in df.columns and delivery_margin > 0:
        mask = mask & (df["DeliveryPct"] >= delivery_margin * df["deliv_avg20"])
        
    return df[mask]

def load_all_data(con, symbols, start_date, end_date, index_df):
    symbols_data = {}
    warmup_days = 100
    start_warm = start_date - timedelta(days=warmup_days)
    
    all_raw = con.execute(
        """
        SELECT symbol, CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
               open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume,
               delivery_pct AS DeliveryPct
        FROM stock_prices
        WHERE symbol != 'NSEI' AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
        ORDER BY symbol, date
        """,
        [start_warm, end_date]
    ).fetchdf()
    
    for symbol, df in all_raw.groupby("symbol"):
        if symbol not in symbols:
            continue
        if df.empty or len(df) < 60:
            continue
            
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        
        # Indicators
        df["sma20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["vol_avg20"] = df["Volume"].shift(1).rolling(20).mean()
        df["rsi14"] = ta.momentum.rsi(df["Close"], window=14)
        
        # Exits
        df["next_open"] = df["Open"].shift(-1)
        df["next_high"] = df["High"].shift(-1)
        df["next_low"] = df["Low"].shift(-1)
        df["next_close"] = df["Close"].shift(-1)
        
        if "DeliveryPct" in df.columns and not df["DeliveryPct"].isnull().all():
            df["deliv_avg20"] = df["DeliveryPct"].shift(1).rolling(20).mean()
        else:
            df["DeliveryPct"] = 0.0
            df["deliv_avg20"] = 0.0
            
        # Circuit Lock filter
        df["close_near_high"] = (df["High"] - df["Close"]) / df["Close"] <= 0.0005
        df["narrow_range"] = (df["High"] - df["Low"]) / df["Close"] < 0.005
        df["circuit_locked"] = df["close_near_high"] & df["narrow_range"]
        
        # Liquidity check
        df["turnover"] = df["Close"] * df["Volume"]
        df["turnover_avg20"] = df["turnover"].shift(1).rolling(20).mean()
        df["liquid"] = df["turnover_avg20"] >= 10000000 # 1 Crore minimum daily turnover
        
        df = df.dropna()
        if df.empty:
            continue
            
        # Derived attributes
        df["pct_today"] = (df["Close"] - df["Open"]) / df["Open"] * 100
        df["vol_ratio"] = df["Volume"] / df["vol_avg20"]
        df["off_high_pct"] = (df["High"] - df["Close"]) / df["High"] * 100
        df["uptrend"] = (df["Close"] > df["sma20"]) & (df["Close"] > df["sma50"])
        df["green_candle"] = df["Close"] > df["Open"]
        df["stock_perf20"] = df["Close"].pct_change(20)
        
        if not index_df.empty:
            df = df.join(index_df[["nifty_close", "nifty_sma20", "nifty_sma50", "nifty_ret5"]], how="left")
            df["nifty_close"] = df["nifty_close"].ffill()
            df["nifty_sma20"] = df["nifty_sma20"].ffill()
            df["nifty_sma50"] = df["nifty_sma50"].ffill()
            df["nifty_ret5"] = df["nifty_ret5"].ffill()
            df["index_sma20_ok"] = df["nifty_close"] > df["nifty_sma20"]
            df["index_sma50_ok"] = df["nifty_close"] > df["nifty_sma50"]
            df["index_perf20"] = df["nifty_close"].pct_change(20)
            df["outperforming_index"] = df["stock_perf20"] > df["index_perf20"]
        else:
            df["index_sma20_ok"] = True
            df["index_sma50_ok"] = True
            df["outperforming_index"] = True
            
        symbols_data[symbol] = df
        
    return symbols_data

def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    con = duckdb.connect(DB_PATH, read_only=True)
    symbols = get_stock_universe(max_stocks=200, no_of_stocks=200)
    symbols = [s for s in symbols if s != 'NSEI']
    
    train_start = date(2021, 7, 22)
    train_end = date(2025, 6, 15)
    test_start = date(2025, 6, 16)
    test_end = date(2026, 7, 22)
    
    logger.info(f"Training Window  : {train_start} -> {train_end}")
    logger.info(f"Validation Window: {test_start} -> {test_end}")
    
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
            index_df["nifty_sma50"] = ta.trend.sma_indicator(index_df["Close"], window=50)
            index_df["nifty_ret5"] = index_df["Close"].pct_change(5)
            index_df = index_df.rename(columns={"Close": "nifty_close"})
    except Exception as e:
        logger.warning(f"Could not load Nifty benchmark: {e}")
        
    logger.info("Loading pricing data for symbols...")
    symbols_data = load_all_data(con, symbols, train_start, test_end, index_df)
    con.close()
    
    near_high_options = [0.008, 0.012, 0.015, 0.020]
    vol_ratio_options = [1.0, 1.25, 1.5]
    min_return_options = [1.0, 1.5, 2.0]
    rsi_ranges = [(50, 75), (55, 78), (60, 80)]
    idx_filters = ["sma50", "sma20"]
    deliv_margins = [0.0, 1.0]
    
    configs = []
    for nh in near_high_options:
        for vr in vol_ratio_options:
            for mr in min_return_options:
                for r_min, r_max in rsi_ranges:
                    for idx_f in idx_filters:
                        for dm in deliv_margins:
                            configs.append({
                                "near_high_pct": nh,
                                "vol_ratio_limit": vr,
                                "min_return": mr,
                                "rsi_min": r_min,
                                "rsi_max": r_max,
                                "index_filter": idx_f,
                                "delivery_margin": dm
                            })
                            
    round_trip_cost = args.cost_pct * 2
    logger.info(f"Sweeping {len(configs)} candidate configurations across {len(symbols_data)} stocks...")
    
    optimized_weights = {}
    
    for symbol, df in symbols_data.items():
        df_train = df[(df.index.date >= train_start) & (df.index.date <= train_end)]
        if df_train.empty:
            continue
            
        best_cfg = None
        best_score = -999.0
        
        for cfg in configs:
            sub_train = get_signals_for_stock(df_train, cfg["near_high_pct"], cfg["vol_ratio_limit"], cfg["min_return"], cfg["rsi_min"], cfg["rsi_max"], cfg["index_filter"], cfg["delivery_margin"])
            n_trades = len(sub_train)
            if n_trades >= args.min_sample_trades:
                rets = compute_exit_type_b_returns(sub_train, round_trip_cost=round_trip_cost)
                win_rate = (rets > 0).sum() / n_trades * 100
                avg_ret = rets.mean()
                
                if win_rate >= args.min_train_win_rate and avg_ret > 0.0:
                    score = win_rate * 5 + avg_ret * 20 + np.sqrt(n_trades)
                    if score > best_score:
                        best_score = score
                        best_cfg = {
                            **cfg,
                            "total_trades": int(n_trades),
                            "win_rate": round(win_rate, 1),
                            "train_avg_ret": round(avg_ret, 2)
                        }
                        
        if best_cfg:
            optimized_weights[symbol] = best_cfg
            
    with open(MODEL_FILE, "w") as f:
        json.dump(optimized_weights, f, indent=4)
        
    logger.info(f"Successfully saved optimized custom BTST parameters for {len(optimized_weights)} stocks to: {MODEL_FILE}")
    
    # Evaluate Out-Of-Sample performance on Holdout Validation period
    test_trades = 0
    test_wins = 0
    test_returns = []
    
    for symbol, cfg in optimized_weights.items():
        df_stock = symbols_data[symbol]
        df_test = df_stock[(df_stock.index.date >= test_start) & (df_stock.index.date <= test_end)]
        sub_test = get_signals_for_stock(df_test, cfg["near_high_pct"], cfg["vol_ratio_limit"], cfg["min_return"], cfg["rsi_min"], cfg["rsi_max"], cfg["index_filter"], cfg["delivery_margin"])
        
        if not sub_test.empty:
            rets = compute_exit_type_b_returns(sub_test, round_trip_cost=round_trip_cost)
            t_count = len(sub_test)
            t_wins = (rets > 0).sum()
            
            test_trades += t_count
            test_wins += t_wins
            test_returns.extend(rets.tolist())
            
    overall_wr = (test_wins / test_trades * 100) if test_trades > 0 else 0.0
    overall_sum_ret = sum(test_returns) if test_returns else 0.0
    overall_avg_ret = np.mean(test_returns) if test_returns else 0.0
    days_in_test = (test_end - test_start).days * (5/7) # trading days approximation (~270)
    trades_per_day = round(test_trades / days_in_test, 2) if days_in_test > 0 else 0.0
    
    logger.info("=" * 80)
    logger.info("=== OVERALL BTST TEST PERFORMANCE (2025-06-16 to 2026-07-22) ===")
    logger.info(f"Qualified Stock Universe Size: {len(optimized_weights)} stocks")
    logger.info(f"Total Test Trades Triggered : {test_trades} (~{trades_per_day} trades/day)")
    logger.info(f"Total Test Wins             : {test_wins}")
    logger.info(f"Overall Test Accuracy (WR)  : {overall_wr:.2f}%")
    logger.info(f"Overall Cumulative Return % : {overall_sum_ret:+.2f}% (Average: {overall_avg_ret:+.2f}% per trade)")
    logger.info("=" * 80)
    return 0

if __name__ == "__main__":
    main()
