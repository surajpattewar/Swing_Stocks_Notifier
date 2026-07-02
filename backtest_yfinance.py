#!/usr/bin/env python3
"""
Backtesting engine using yfinance historical data.
Evaluates individual indicators (pointers) and their combinations.
Saves detailed and summary reports in CSV format.
"""

import os
import sys
import logging
import argparse
import itertools
import pickle
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("backtest_yfinance")

# Add current directory to path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from screener import evaluate, Candidate
from stock_universe import get_stock_universe

CACHE_PATH = "data/stock_history_cache.pkl"

def load_cached_data(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, float], list[str]]:
    """
    Load cached data for matching symbols.
    Returns:
        loaded_stock_data: dict of symbol -> DataFrame
        loaded_betas: dict of symbol -> float
        missing_symbols: list of symbols not found in the cache
    """
    loaded_stock_data = {}
    loaded_betas = {}
    missing_symbols = []
    
    if os.path.exists(CACHE_PATH):
        try:
            logger.info(f"Loading cached stock history from {CACHE_PATH}...")
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
                cached_data = cache.get("stock_data", {})
                cached_betas = cache.get("betas", {})
                
                for sym in symbols:
                    if sym in cached_data:
                        loaded_stock_data[sym] = cached_data[sym]
                        loaded_betas[sym] = cached_betas.get(sym, 0.0)
                    else:
                        missing_symbols.append(sym)
                logger.info(f"Successfully loaded cache for {len(loaded_stock_data)} symbols.")
                return loaded_stock_data, loaded_betas, missing_symbols
        except Exception as e:
            logger.error(f"Failed to read cache file: {e}")
            
    return {}, {}, symbols

def save_cached_data(stock_data: dict[str, pd.DataFrame], betas: dict[str, float]):
    """Save/update the local cache file with fetched stock data and betas."""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        existing_data = {}
        existing_betas = {}
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "rb") as f:
                    cache = pickle.load(f)
                    existing_data = cache.get("stock_data", {})
                    existing_betas = cache.get("betas", {})
            except Exception:
                pass
                
        # Merge
        existing_data.update(stock_data)
        existing_betas.update(betas)
        
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({
                "stock_data": existing_data,
                "betas": existing_betas,
                "timestamp": datetime.now()
            }, f)
        logger.info(f"Saved cache for {len(existing_data)} total symbols to {CACHE_PATH}")
    except Exception as e:
        logger.error(f"Failed to write cache file: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="yfinance multi-pointer backtester.")
    parser.add_argument("--years", type=float, default=5.0,
                        help="Number of years of historical data to backtest (default: 5.0)")
    parser.add_argument("--min-score", type=int, default=1,
                        help="Minimum score to save a signal (default: 1, to test all pointers)")
    parser.add_argument("--max-holding-days", type=int, default=15,
                        help="Maximum trading days to hold a position (default: 15)")
    parser.add_argument("--max-stocks", type=int, default=50,
                        help="Maximum stocks to scan from universe (default: 50)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols to test instead of universe")
    parser.add_argument("--output-dir", type=str, default="backtest_results",
                        help="Directory to save output CSV files (default: backtest_results)")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Parallel workers for downloading/processing (default: 10)")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached stock history if available")
    parser.add_argument("--save-cache", action="store_true",
                        help="Save downloaded stock history to cache")
    return parser.parse_args()

def fetch_stock_data(symbol: str, start_date: date, end_date: date) -> tuple[str, pd.DataFrame | None, float]:
    """Fetch history and beta from yfinance for a single stock."""
    try:
        logger.info(f"Downloading history & info for {symbol}...")
        ticker = yf.Ticker(symbol)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        df = ticker.history(start=start_str, end=end_str, interval="1d", auto_adjust=True)
        
        if df.empty:
            logger.info(f"Start/end query returned empty for {symbol}, falling back to period='6y'")
            df = ticker.history(period="6y", interval="1d", auto_adjust=True)
            
        if df.empty or len(df) < 130:
            logger.warning(f"Insufficient history for {symbol} (len={len(df)})")
            return symbol, None, 0.0
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Get beta (default to 0.0 on error/missing)
        beta = 0.0
        try:
            beta = float(ticker.info.get("beta") or 0.0)
        except Exception:
            pass
            
        return symbol, df, beta
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return symbol, None, 0.0

def simulate_outcome(df: pd.DataFrame, signal_date, entry_price: float,
                      stop_loss: float, target: float, max_holding_days: int) -> dict:
    """
    Look forward from signal_date (exclusive) up to max_holding_days to check target/stop hits.
    Conservatively treats same-day ties as stop-loss hits first.
    """
    future = df.loc[df.index > signal_date].iloc[:max_holding_days]

    if future.empty:
        return {"outcome": "no_data", "exit_price": None, "exit_date": None,
                "days_held": 0, "return_pct": None}

    for i, (dt, row) in enumerate(future.iterrows(), start=1):
        hit_target = row["High"] >= target
        hit_stop = row["Low"] <= stop_loss
        if hit_stop:  # Conservative: stop loss hit wins tie
            exit_price = stop_loss
            return {"outcome": "stop_loss_hit", "exit_price": exit_price,
                     "exit_date": dt.date(), "days_held": i,
                     "return_pct": round((exit_price - entry_price) / entry_price * 100, 2)}
        if hit_target:
            exit_price = target
            return {"outcome": "target_hit", "exit_price": exit_price,
                     "exit_date": dt.date(), "days_held": i,
                     "return_pct": round((exit_price - entry_price) / entry_price * 100, 2)}

    # Timeout exit
    last_row = future.iloc[-1]
    exit_price = float(last_row["Close"])
    return {"outcome": "timeout", "exit_price": exit_price,
             "exit_date": future.index[-1].date(), "days_held": len(future),
             "return_pct": round((exit_price - entry_price) / entry_price * 100, 2)}

def process_symbol(symbol: str, df: pd.DataFrame, beta: float, backtest_start: date, 
                   backtest_end: date, min_score: int, max_holding_days: int) -> list[dict]:
    """Perform walk-forward simulation for a single symbol."""
    results = []
    # Find all trading dates in backtest window
    eval_dates = df.index[(df.index >= pd.Timestamp(backtest_start)) & (df.index <= pd.Timestamp(backtest_end))]
    
    for eval_date in eval_dates:
        df_slice = df.loc[:eval_date]
        if len(df_slice) < 130:
            continue
            
        # Take the last 130 rows to pass to evaluate for indicator calculation
        df_slice_warm = df_slice.iloc[-130:]
        
        try:
            cand = evaluate(symbol, df_slice_warm, {"beta": beta})
        except Exception:
            continue
            
        if cand.score < min_score:
            continue
            
        # Simulate trade outcome
        outcome = simulate_outcome(df, eval_date, cand.close, cand.stop_loss, cand.target, max_holding_days)
        
        row = {
            "symbol": symbol,
            "signal_date": eval_date.strftime("%Y-%m-%d"),
            "score": cand.score,
            "setup_type": cand.setup_type,
            "entry_price": cand.close,
            "rsi": cand.rsi,
            "adx": cand.adx,
            "stop_loss": cand.stop_loss,
            "target": cand.target,
            "reasons": "; ".join(cand.reasons),
            "outcome": outcome["outcome"],
            "exit_price": outcome["exit_price"],
            "exit_date": outcome["exit_date"].strftime("%Y-%m-%d") if outcome["exit_date"] else None,
            "days_held": outcome["days_held"],
            "return_pct": outcome["return_pct"]
        }
        
        # Inject individual pointer flags dynamically
        for sig_name, sig_val in cand.signals.items():
            row[f"pointer_{sig_name}"] = sig_val
            
        results.append(row)
        
    return results

def compute_subset_metrics(subset: pd.DataFrame, combination_name: str) -> dict | None:
    """Calculate performance metrics for a filtered subset of signals."""
    total_signals = len(subset)
    if total_signals == 0:
        return None
        
    closed = subset[subset["outcome"] != "no_data"]
    decided = closed[closed["outcome"].isin(["target_hit", "stop_loss_hit"])]
    timeouts = closed[closed["outcome"] == "timeout"]
    
    n_decided = len(decided)
    n_wins = (decided["outcome"] == "target_hit").sum()
    n_losses = (decided["outcome"] == "stop_loss_hit").sum()
    
    win_rate_decided = round(100 * n_wins / n_decided, 1) if n_decided > 0 else None
    
    closed_for_pct = closed.dropna(subset=["return_pct"])
    n_profitable = (closed_for_pct["return_pct"] > 0).sum()
    win_rate_overall = round(100 * n_profitable / len(closed_for_pct), 1) if len(closed_for_pct) > 0 else None
    
    avg_return = round(closed_for_pct["return_pct"].mean(), 2) if len(closed_for_pct) > 0 else None
    
    gross_gain = closed_for_pct.loc[closed_for_pct["return_pct"] > 0, "return_pct"].sum()
    gross_loss = -closed_for_pct.loc[closed_for_pct["return_pct"] < 0, "return_pct"].sum()
    
    profit_factor = round(gross_gain / gross_loss, 2) if gross_loss > 0 else (99.9 if gross_gain > 0 else 1.0)
    if gross_loss == 0 and gross_gain == 0:
        profit_factor = 1.0
        
    return {
        "combination": combination_name,
        "total_signals": total_signals,
        "closed_signals": len(closed),
        "target_hits": int(n_wins),
        "stop_loss_hits": int(n_losses),
        "timeouts": len(timeouts),
        "win_rate_target_vs_stop_pct": win_rate_decided,
        "win_rate_overall_pct": win_rate_overall,
        "avg_return_pct": avg_return,
        "profit_factor": profit_factor
    }

def analyze_permutations(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze combinations of pointers and find top-performing setups."""
    if signals_df.empty:
        return pd.DataFrame()
        
    pointer_cols = [c for c in signals_df.columns if c.startswith("pointer_")]
    if not pointer_cols:
        return pd.DataFrame()
        
    results = []
    
    # 1. Individual pointers
    for col in pointer_cols:
        pointer_name = col.replace("pointer_", "")
        subset = signals_df[signals_df[col] == True]
        metrics = compute_subset_metrics(subset, f"Single: {pointer_name}")
        if metrics:
            results.append(metrics)
            
    # 2. Pairs of pointers
    for col1, col2 in itertools.combinations(pointer_cols, 2):
        name1 = col1.replace("pointer_", "")
        name2 = col2.replace("pointer_", "")
        subset = signals_df[(signals_df[col1] == True) & (signals_df[col2] == True)]
        metrics = compute_subset_metrics(subset, f"{name1} + {name2}")
        if metrics:
            results.append(metrics)
            
    # 3. Triplets of pointers (only combinations with at least 5 signals to avoid noise)
    for col1, col2, col3 in itertools.combinations(pointer_cols, 3):
        name1 = col1.replace("pointer_", "")
        name2 = col2.replace("pointer_", "")
        name3 = col3.replace("pointer_", "")
        subset = signals_df[(signals_df[col1] == True) & (signals_df[col2] == True) & (signals_df[col3] == True)]
        if len(subset) >= 5:
            metrics = compute_subset_metrics(subset, f"{name1} + {name2} + {name3}")
            if metrics:
                results.append(metrics)
            
    # 4. Score thresholds
    for min_score in [3, 4, 5, 6]:
        subset = signals_df[signals_df["score"] >= min_score]
        metrics = compute_subset_metrics(subset, f"Score >= {min_score}")
        if metrics:
            results.append(metrics)
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Sort by profit factor and win rate
        res_df["sort_pf"] = res_df["profit_factor"].fillna(0.0)
        res_df = res_df.sort_values(by=["sort_pf", "win_rate_target_vs_stop_pct"], ascending=False).drop(columns=["sort_pf"])
        
    return res_df

def print_table(title: str, df: pd.DataFrame, max_rows: int = 15):
    """Utility to print dataframes in a neat text table format."""
    print("\n" + "=" * 90)
    print(f" {title.upper()} (Top {max_rows} rows)")
    print("=" * 90)
    if df.empty:
        print("No data available.")
        return
        
    display_df = df.head(max_rows).copy()
    headers = list(display_df.columns)
    rows = []
    for _, r in display_df.iterrows():
        row_vals = []
        for h in headers:
            val = r[h]
            if isinstance(val, float):
                row_vals.append(f"{val:.2f}")
            elif val is None:
                row_vals.append("N/A")
            else:
                row_vals.append(str(val))
        rows.append(row_vals)
        
    widths = [len(h) for h in headers]
    for r in rows:
        for idx, val in enumerate(r):
            widths[idx] = max(widths[idx], len(val))
            
    header_str = " | ".join(f"{h:<{widths[idx]}}" for idx, h in enumerate(headers))
    separator = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    
    print(header_str)
    print(separator)
    for r in rows:
        row_str = " | ".join(f"{val:<{widths[idx]}}" for idx, val in enumerate(r))
        print(row_str)
    print("=" * 90 + "\n")

def main():
    args = parse_args()
    
    # 1. Build stock list
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        # Load from stock_universe (limit the amount of downloaded tickers to avoid timeouts)
        logger.info(f"Loading stock universe list (capping to {args.max_stocks} stocks)...")
        symbols = get_stock_universe(max_stocks=args.max_stocks)
        
    # Clean up index symbols
    symbols = [s for s in symbols if not s.startswith("^")]
    
    if not symbols:
        logger.error("No valid symbols to backtest.")
        return 1
        
    logger.info(f"Starting backtest for {len(symbols)} symbols over the last {args.years} years...")
    
    # Define backtest window
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=int(args.years * 365))
    
    # Warmup buffer (130 trading days is ~200 calendar days)
    warmup_start_date = start_date - timedelta(days=200)
    
    # 2. Fetch stock data using cache / yf.download in a single batch
    stock_data = {}
    betas = {}
    symbols_to_download = list(symbols)
    
    if args.use_cache:
        cached_data, cached_betas, missing = load_cached_data(symbols)
        stock_data.update(cached_data)
        betas.update(cached_betas)
        symbols_to_download = missing
        if not symbols_to_download:
            logger.info("All requested symbols loaded from cache. Skipping downloads.")
            
    if symbols_to_download:
        logger.info(f"Fetching historical prices from {warmup_start_date} to {end_date} in batch for {len(symbols_to_download)} symbols...")
        try:
            start_str = warmup_start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            if len(symbols_to_download) == 1:
                df_all = yf.download(symbols_to_download[0], start=start_str, end=end_str, interval="1d", auto_adjust=True, progress=False)
                if not df_all.empty:
                    if df_all.index.tz is not None:
                        df_all.index = df_all.index.tz_localize(None)
                    stock_data[symbols_to_download[0]] = df_all
            else:
                df_all = yf.download(symbols_to_download, start=start_str, end=end_str, interval="1d", auto_adjust=True, group_by='ticker', progress=False)
                for sym in symbols_to_download:
                    if sym in df_all:
                        df_sym = df_all[sym].dropna(how='all')
                        if not df_sym.empty and len(df_sym) >= 130:
                            if df_sym.index.tz is not None:
                                df_sym.index = df_sym.index.tz_localize(None)
                            stock_data[sym] = df_sym
        except Exception as e:
            logger.error(f"Failed to batch download stock history: {e}")
            
    if not stock_data:
        logger.error("No data fetched. Exiting.")
        return 1
        
    # Politely fetch missing betas
    missing_betas_symbols = [sym for sym in stock_data.keys() if sym not in betas]
    if missing_betas_symbols:
        logger.info(f"Politely fetching beta info for {len(missing_betas_symbols)} active symbols...")
        import time
        for idx, sym in enumerate(missing_betas_symbols, 1):
            beta = 0.0
            try:
                ticker = yf.Ticker(sym)
                beta = float(ticker.info.get("beta") or 0.0)
            except Exception:
                pass
            betas[sym] = beta
            time.sleep(0.1)  # small pause
            if idx % 10 == 0 or idx == len(missing_betas_symbols):
                logger.info(f"Beta info progress: {idx}/{len(missing_betas_symbols)} symbols processed.")
                
    # Save cache if requested
    if args.save_cache and stock_data:
        save_cached_data(stock_data, betas)
        
    # 3. Process walk-forward logic in parallel
    all_signals = []
    logger.info("Running walk-forward backtest simulation...")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for sym, df in stock_data.items():
            beta = betas[sym]
            futures[executor.submit(process_symbol, sym, df, beta, start_date, end_date, args.min_score, args.max_holding_days)] = sym
            
        for idx, fut in enumerate(as_completed(futures), start=1):
            sym = futures[fut]
            try:
                res = fut.result()
                all_signals.extend(res)
            except Exception as e:
                logger.error(f"Error processing {sym}: {e}")
            if idx % 10 == 0 or idx == len(stock_data):
                logger.info(f"Simulation progress: {idx}/{len(stock_data)} tickers processed.")
                
    # 4. Process results
    signals_df = pd.DataFrame(all_signals)
    
    if signals_df.empty:
        logger.warning("No trade signals generated by the strategy within this window.")
        return 0
        
    signals_df = signals_df.sort_values(by=["signal_date", "symbol"]).reset_index(drop=True)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed signal records
    signals_file = os.path.join(args.output_dir, f"backtest_signals_{timestamp}.csv")
    signals_df.to_csv(signals_file, index=False)
    logger.info(f"Detailed signals written to: {signals_file}")
    
    # Run permutation analysis
    logger.info("Analyzing individual pointer accuracies and permutations...")
    analysis_df = analyze_permutations(signals_df)
    
    # Save summary report
    summary_file = os.path.join(args.output_dir, f"backtest_summary_{timestamp}.csv")
    analysis_df.to_csv(summary_file, index=False)
    logger.info(f"Summary metrics written to: {summary_file}")
    
    # Print nice formatted console report
    print("\n" + "=" * 90)
    print("                      SWING SCREENER BACKTEST REPORT (yfinance)")
    print("=" * 90)
    print(f"Backtest period : {start_date} to {end_date} ({args.years} years)")
    print(f"Total symbols    : {len(stock_data)}")
    print(f"Total signals    : {len(signals_df)}")
    print(f"Max hold period  : {args.max_holding_days} trading days")
    print("=" * 90)
    
    # Extract single pointers for a quick summary
    single_pointers = analysis_df[analysis_df["combination"].str.startswith("Single:")].copy()
    print_table("Individual Pointer Performance", single_pointers)
    
    # Extract top combinations (excluding single pointers)
    combinations = analysis_df[~analysis_df["combination"].str.startswith("Single:")].copy()
    # Filter for combinations with at least 5 signals to avoid noise
    valid_combos = combinations[combinations["total_signals"] >= 5]
    print_table("Top Combined Pointers (Min 5 signals)", valid_combos, max_rows=15)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
