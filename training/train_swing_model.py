#!/usr/bin/env python3

# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import ta
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



from config import config
from backtesting.backtest import get_local_symbols, get_latest_price_date
from screener import fetch_history, _add_indicators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_swing_model")

DB_PATH = config.DUCKDB_PATH
workspace_dir = _parent_dir
MODEL_DIR = os.path.join(workspace_dir, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "swing_stock_weights.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize swing trading parameters per stock for setup classes.")
    parser.add_argument("--train-months", type=int, default=48,
                        help="How many months of history to train on (default: 48)")
    parser.add_argument("--min-signals", type=int, default=5,
                        help="Minimum signals required to qualify a configuration (default: 5)")
    parser.add_argument("--max-stocks", type=int, default=250,
                        help="Maximum stocks to process (default: 250)")
    parser.add_argument("--target-accuracy", type=float, default=80.0,
                        help="Target accuracy threshold in percent (default: 80.0)")
    return parser.parse_args()

def fetch_long_history(symbol, months, max_holding_days=15, db_path=None, as_of_date=None):
    try:
        import duckdb
        with duckdb.connect(db_path, read_only=True) as con:
            df = con.execute(
                """
                SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                       open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
                FROM stock_prices
                WHERE symbol = ? AND CAST(timezone('Asia/Kolkata', date) AS DATE) <= ?
                ORDER BY date
                """,
                [symbol, as_of_date]
            ).fetchdf()
            if df.empty:
                return pd.DataFrame()
            df["Date"] = pd.to_datetime(df["Date"])
            return df.set_index("Date")
    except Exception as e:
        logger.warning("DuckDB load error for %s: %s", symbol, e)
    return pd.DataFrame()

def simulate_trade_outcome_numpy(lows, highs, closes, entry_pos, entry_price, stop_loss, target, max_holding_days=15):
    n = len(lows)
    for offset in range(1, max_holding_days + 1):
        idx = entry_pos + offset
        if idx >= n:
            break
        h = highs[idx]
        l = lows[idx]
        c = closes[idx]
        if l <= stop_loss and h >= target:
            return -0.5  # conservative drag/loss
        if l <= stop_loss:
            return (stop_loss - entry_price) / entry_price * 100
        if h >= target:
            return (target - entry_price) / entry_price * 100
    exit_c = closes[min(entry_pos + max_holding_days, n - 1)]
    return (exit_c - entry_price) / entry_price * 100

def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    symbols = get_local_symbols(DB_PATH, max_stocks=args.max_stocks)
    symbols = [s for s in symbols if s != 'NSEI']
    if not symbols:
        logger.error(f"No symbols found in price table at {DB_PATH}")
        return 1

    # Load Nifty index data
    import duckdb
    index_df = None
    try:
        with duckdb.connect(DB_PATH, read_only=True) as con:
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
                index_df["sma20"] = ta.trend.sma_indicator(index_df["Close"], window=20)
                index_df["sma50"] = ta.trend.sma_indicator(index_df["Close"], window=50)
    except Exception as e:
        logger.warning(f"Could not load Nifty benchmark: {e}")

    # Validation splits preventing holdout data leakage
    train_end_date = pd.Timestamp("2025-06-15").date()
    test_start_date = pd.Timestamp("2025-06-16").date()
    latest_date = pd.Timestamp("2026-01-22").date()
    
    start_date = train_end_date - timedelta(days=int(args.train_months * 31))
    logger.info(f"Training Window  : {start_date} -> {train_end_date}")
    logger.info(f"Validation Window: {test_start_date} -> {latest_date}")

    TREND_FEATURES = ["uptrend_sma50", "uptrend_sma200", "adx_strong_trend", "ema_cross", "volume_spike", "bullish_engulfing"]
    PULLBACK_FEATURES = ["bb_pullback", "rsi2_pullback", "rsi14_oversold", "stoch_d_turn"]

    sl_atr_options = [1.0, 1.2, 1.5, 1.8]
    target_atr_options = [1.5, 1.8, 2.0, 2.5, 3.0]

    optimized_configs = {}
    total_trained = 0
    total_test_trades = 0
    total_test_wins = 0
    total_test_return_sum = 0.0

    for idx, symbol in enumerate(symbols, 1):
        try:
            df_raw = fetch_long_history(symbol, args.train_months + 2, max_holding_days=15, db_path=DB_PATH, as_of_date=latest_date)
            if df_raw.empty or len(df_raw) < 220:
                continue

            df = _add_indicators(df_raw.copy())
            if df.empty or len(df) < 50:
                continue

            # Merge Index trend
            if index_df is not None and not index_df.empty:
                df = df.join(index_df.rename(columns={"Close": "index_close", "sma20": "index_sma20", "sma50": "index_sma50"}), how="left")
                df["index_close"] = df["index_close"].ffill()
                df["index_sma20"] = df["index_sma20"].ffill()
                df["index_sma50"] = df["index_sma50"].ffill()
            else:
                df["index_close"] = df["Close"]
                df["index_sma20"] = df["Close"]
                df["index_sma50"] = df["Close"]

            # Vectorized setup class indicator calculations
            df["uptrend_sma50"] = (df["Close"] > df["sma50"]) & (df["sma50"] > df["sma50"].shift(5))
            df["uptrend_sma200"] = df["Close"] > df["sma200"]
            df["adx_strong_trend"] = df["adx"] >= 25
            df["ema_cross"] = df["ema5"] > df["ema20"]
            df["volume_spike"] = df["Volume"] > 1.5 * df["vol_avg20"]
            df["bb_pullback"] = df["Low"] <= df["bb_low"]
            df["rsi2_pullback"] = df["rsi2"] < 5
            df["rsi14_oversold"] = df["rsi14"] < 35
            df["stoch_d_turn"] = df["stoch_d"] > df["stoch_d"].shift(1)

            eval_dates_train = df.index[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(train_end_date))]
            train_df = df.loc[eval_dates_train]

            lows_arr = df_raw["Low"].values.astype(np.float64)
            highs_arr = df_raw["High"].values.astype(np.float64)
            closes_arr = df_raw["Close"].values.astype(np.float64)

            stock_config = {}

            # Train both setup classes separately
            for sc, features, max_thresh in [("trend", TREND_FEATURES, 5.0), ("pullback", PULLBACK_FEATURES, 3.5)]:
                best_score = -1.0
                best_cfg = None

                # Pre-gather training signals
                candidates = []
                for signal_date in train_df.index:
                    row = df.loc[signal_date]
                    if row["index_close"] <= row["index_sma50"]:
                        continue # Skip weak market entries in training (using sma50 is more lenient)
                        
                    future_bars = df_raw.loc[df_raw.index > signal_date]
                    if future_bars.empty:
                        continue
                    next_open = float(future_bars.iloc[0]["Open"])
                    entry_pos = df_raw.index.get_loc(future_bars.index[0])
                    
                    candidates.append({
                        "entry_pos": int(entry_pos),
                        "entry_price": float(next_open),
                        "atr": float(row["atr"]),
                        "close": float(row["Close"]),
                        "signals": {f: bool(row[f]) for f in features}
                    })

                if len(candidates) < args.min_signals:
                    continue

                X_all = np.array([[float(c["signals"][f]) for f in features] for c in candidates])

                # Grid Search stop loss and target ATR multipliers
                for sl_atr in sl_atr_options:
                    for target_atr in target_atr_options:
                        y_outcomes = []
                        valid_indices = []
                        
                        for c_idx, c in enumerate(candidates):
                            atr_pct = c["atr"] / c["entry_price"]
                            target_pct = atr_pct * target_atr
                            sl_pct = atr_pct * sl_atr
                            
                            sl_pct = max(sl_pct, 0.5 * atr_pct)
                            target_pct = min(target_pct, 0.12)
                            
                            target_price = round(c["entry_price"] * (1.0 + target_pct), 2)
                            stop_loss_price = round(c["entry_price"] * (1.0 - sl_pct), 2)
                            
                            if c["entry_price"] >= target_price:
                                continue
                                
                            ret = simulate_trade_outcome_numpy(lows_arr, highs_arr, closes_arr, c["entry_pos"], c["entry_price"], stop_loss_price, target_price)
                            y_outcomes.append(1 if ret > 0 else 0)
                            valid_indices.append(c_idx)
                            
                        if len(y_outcomes) < args.min_signals:
                            continue
                            
                        y = np.array(y_outcomes)
                        X = X_all[valid_indices]
                        
                        if len(np.unique(y)) < 2:
                            continue
                            
                        clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=42)
                        clf.fit(X, y)
                        probs = clf.predict_proba(X)[:, 1]
                        
                        # Threshold search
                        for thresh in np.arange(0.5, 0.91, 0.05):
                            mask = probs >= thresh
                            triggered = mask.sum()
                            if triggered < args.min_signals:
                                continue
                                
                            wins = y[mask].sum()
                            win_rate = (wins / triggered) * 100
                            
                            expectancy = (win_rate / 100.0) * target_atr - ((100.0 - win_rate) / 100.0) * sl_atr
                            if expectancy <= 0.1:
                                continue
                                
                            if win_rate >= args.target_accuracy:
                                coefs = clf.coef_[0]
                                weights_map = {f_name: max(0.1, float(coef)) for f_name, coef in zip(features, coefs)}
                                max_coef = max(weights_map.values()) if weights_map else 0.0
                                
                                if max_coef > 0.0:
                                    scale_factor = 2.0 / max_coef
                                    logit_thresh = np.log(thresh / (1 - thresh))
                                    raw_threshold = logit_thresh - clf.intercept_[0]
                                    scaled_threshold = round(scale_factor * raw_threshold, 1)
                                    min_bound = 2.5 if sc == "trend" else 1.8
                                    scaled_threshold = max(min(scaled_threshold, max_thresh), min_bound)
                                    
                                    scaled_weights = {k: round(v * scale_factor, 1) for k, v in weights_map.items()}
                                    
                                    # Score: net profit expectancy & trade volume priority
                                    score = expectancy * 1000 + win_rate * 10 + triggered * 0.5
                                    if score > best_score:
                                        best_score = score
                                        best_cfg = {
                                            "weights": scaled_weights,
                                            "stop_loss_atr": float(sl_atr),
                                            "target_atr": float(target_atr),
                                            "min_score": float(scaled_threshold),
                                            "win_rate": round(win_rate, 1),
                                            "total_trades": int(triggered)
                                        }

                if best_cfg:
                    # Evaluate on Out-Of-Sample validation set
                    eval_dates_test = df.index[(df.index >= pd.Timestamp(test_start_date)) & (df.index <= pd.Timestamp(latest_date))]
                    test_trades = 0
                    test_wins = 0
                    test_return_sum = 0.0
                    
                    for signal_date in eval_dates_test:
                        row = df.loc[signal_date]
                        if row["index_close"] <= row["index_sma50"]:
                            continue
                            
                        # Evaluate score
                        score = sum(best_cfg["weights"].get(f, 1.0) * float(row[f]) for f in features)
                        if score >= best_cfg["min_score"]:
                            future_bars = df_raw.loc[df_raw.index > signal_date]
                            if future_bars.empty:
                                continue
                            next_open = float(future_bars.iloc[0]["Open"])
                            entry_pos = df_raw.index.get_loc(future_bars.index[0])
                            
                            atr_val = float(row["atr"])
                            atr_pct = atr_val / next_open
                            target_pct = atr_pct * best_cfg["target_atr"]
                            sl_pct = atr_pct * best_cfg["stop_loss_atr"]
                            
                            sl_pct = max(sl_pct, 0.5 * atr_pct)
                            target_pct = min(target_pct, 0.12)
                            
                            target_price = round(next_open * (1.0 + target_pct), 2)
                            stop_loss_price = round(next_open * (1.0 - sl_pct), 2)
                            
                            if next_open >= target_price:
                                continue
                                
                            ret = simulate_trade_outcome_numpy(lows_arr, highs_arr, closes_arr, entry_pos, next_open, stop_loss_price, target_price)
                            test_trades += 1
                            if ret > 0:
                                test_wins += 1
                            test_return_sum += ret

                    best_cfg["test_trades"] = test_trades
                    best_cfg["test_wins"] = test_wins
                    best_cfg["test_return_sum"] = round(test_return_sum, 2)
                    best_cfg["test_win_rate"] = round((test_wins / test_trades * 100), 1) if test_trades > 0 else 0.0
                    
                    total_test_trades += test_trades
                    total_test_wins += test_wins
                    total_test_return_sum += test_return_sum
                    stock_config[sc] = best_cfg

            if stock_config:
                stock_config["total_trades"] = max(stock_config.get("trend", {}).get("total_trades", 0), stock_config.get("pullback", {}).get("total_trades", 0))
                optimized_configs[symbol] = stock_config
                total_trained += 1
                logger.info(f"[{total_trained}] Optimized {symbol} for Trend ({stock_config.get('trend', {}).get('win_rate', 0.0)}% win rate) & Pullback ({stock_config.get('pullback', {}).get('win_rate', 0.0)}% win rate)")

        except Exception as e:
            logger.exception(f"Error optimizing {symbol}: {e}")
            continue

    # Save weights
    with open(MODEL_FILE, "w") as f:
        json.dump(optimized_configs, f, indent=4)

    logger.info(f"Completed! Custom weights targeting >={args.target_accuracy}% win rate saved for {total_trained}/{len(symbols)} stocks to: {MODEL_FILE}")
    
    overall_test_win_rate = (total_test_wins / total_test_trades * 100) if total_test_trades > 0 else 0.0
    overall_test_avg_return = (total_test_return_sum / total_test_trades) if total_test_trades > 0 else 0.0
    logger.info(f"=== OVERALL SWING TEST PERFORMANCE (2025-06-16 to 2026-01-22) ===")
    logger.info(f"Total Test Trades Triggered: {total_test_trades}")
    logger.info(f"Total Test Wins            : {total_test_wins}")
    logger.info(f"Overall Test Accuracy (WR) : {overall_test_win_rate:.2f}%")
    logger.info(f"Overall Test Return %      : {total_test_return_sum:.2f}% (Average: {overall_test_avg_return:+.2f}% per trade)")
    return 0

if __name__ == "__main__":
    main()