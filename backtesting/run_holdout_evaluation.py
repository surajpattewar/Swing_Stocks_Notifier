#!/usr/bin/env python3

# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
import os
import sys
import duckdb
import pandas as pd
from datetime import date, timedelta



from config import config
from backtest import run_backtest, compute_accuracy_metrics
from run_btst_backtest import run_btst_backtest, load_data, get_symbols

def main():
    db_path = config.DUCKDB_PATH
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return 1

    con = duckdb.connect(db_path, read_only=True)
    symbols = get_symbols(con)
    
    # Define out-of-sample holdout window
    holdout_start = date(2026, 1, 23)
    holdout_end = date(2026, 7, 22)
    
    print("\n" + "=" * 80)
    # ----------------- Benchmark loading -----------------
    index_df = None
    try:
        index_raw = con.execute(
            """
            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                   close AS Close
            FROM stock_prices
            WHERE symbol = 'NSEI' AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
            ORDER BY date
            """,
            [holdout_start - timedelta(days=100), holdout_end]
        ).fetchdf()
        if not index_raw.empty:
            index_raw["Date"] = pd.to_datetime(index_raw["Date"])
            index_df = index_raw.set_index("Date")
    except Exception as e:
        print(f"Warning: Could not load index benchmark: {e}")

    # 1. Run Swing backtest (Global Scoring Weights, min_score = 5)
    print(f"Running SWING holdout backtest (min_score=5, {holdout_start} to {holdout_end})...")
    swing_results = run_backtest(
        symbols=symbols,
        min_score=5,
        start_date=holdout_start,
        end_date=holdout_end,
        max_workers=4
    )
    
    # 2. Run BTST backtest (delivery_margin = 1.25x)
    print(f"Running BTST holdout backtest (margin=1.25x, {holdout_start} to {holdout_end})...")
    symbols_data = {}
    for sym in symbols:
        symbols_data[sym] = load_data(con, sym, holdout_start, holdout_end)
        
    btst_results = run_btst_backtest(
        symbols_data=symbols_data,
        start_date=holdout_start,
        end_date=holdout_end,
        index_df=index_df,
        delivery_margin=1.25
    )
    
    con.close()
    
    # Summarize Swing results
    swing_metrics = compute_accuracy_metrics(swing_results)
    
    # Summarize BTST results (Limit Exit Type B)
    btst_metrics = {}
    if not btst_results.empty:
        wins_limit = (btst_results["ret_limit"] > 0).sum()
        total_btst = len(btst_results)
        decided_btst = total_btst
        win_rate_btst = wins_limit / total_btst * 100 if total_btst else 0.0
        avg_ret_btst = btst_results["ret_limit"].mean() if total_btst else 0.0
        gross_win = btst_results.loc[btst_results["ret_limit"] > 0, "ret_limit"].sum()
        gross_loss = -btst_results.loc[btst_results["ret_limit"] < 0, "ret_limit"].sum()
        pf_btst = gross_win / gross_loss if gross_loss > 0 else 99.9
        gap_downs = (btst_results["ret_open_raw"] < 0).sum()
        fpr = gap_downs / total_btst * 100 if total_btst else 0.0
        
        btst_metrics = {
            "total_signals": total_btst,
            "win_rate": win_rate_btst,
            "avg_return": avg_ret_btst,
            "profit_factor": pf_btst,
            "false_pos_rate": fpr
        }

    overlap_df = pd.DataFrame()
    if not swing_results.empty and not btst_results.empty:
        swing_results["signal_date"] = pd.to_datetime(swing_results["signal_date"])
        btst_results["date"] = pd.to_datetime(btst_results["date"])
        
        # Merge on symbol and date
        overlap_df = pd.merge(
            swing_results[["symbol", "signal_date", "score", "setup_class"]],
            btst_results[["symbol", "date", "ret_limit"]],
            left_on=["symbol", "signal_date"],
            right_on=["symbol", "date"]
        )

    # ----------------- PRINT RESULTS -----------------
    print("\n" + "=" * 80)
    print("                 STRICT OUT-OF-SAMPLE HOLDOUT EVALUATION")
    print("=" * 80)
    print(f"Period: {holdout_start} to {holdout_end}")
    
    print("\n[SWING PERFORMANCE]")
    print(f"  Total Signals Generated      : {swing_metrics.get('total_signals', 0)}")
    print(f"  Decided Trades (T/SL hit)    : {swing_metrics.get('decided_signals', 0)}")
    print(f"  Win Rate (Target vs SL)      : {swing_metrics.get('win_rate_target_vs_stop_pct', 'N/A')}%")
    print(f"  Average Return per Signal    : {swing_metrics.get('avg_return_pct_per_signal', 'N/A')}%")
    print(f"  Profit Factor (gross return) : {swing_metrics.get('profit_factor', 'N/A')}")
    
    for sc in ["trend", "pullback"]:
        if f"class_{sc}" in swing_metrics:
            cm = swing_metrics[f"class_{sc}"]
            print(f"    - Setup Class {sc.upper()}: WinRate={cm['win_rate_target_vs_stop_pct']}% | Return={cm['avg_return_pct_per_signal']}% | ProfitFactor={cm['profit_factor']}")

    print("\n[BTST PERFORMANCE (1.25x Margin Filter)]")
    if btst_metrics:
        print(f"  Total Signals Generated      : {btst_metrics['total_signals']}")
        print(f"  Win Rate (Target vs SL)      : {btst_metrics['win_rate']:.2f}%")
        print(f"  Average Return per Signal    : {btst_metrics['avg_return']:+.2f}%")
        print(f"  Profit Factor (gross return) : {btst_metrics['profit_factor']:.2f}")
        print(f"  False Positive Rate          : {btst_metrics['false_pos_rate']:.2f}%")
    else:
        print("  No signals generated in holdout period.")

    print("\n[SWING & BTST SIGNAL OVERLAP ANALYSIS]")
    if not overlap_df.empty:
        print(f"  Number of overlapping signals: {len(overlap_df)}")
        print("  Overlapping Signals details:")
        print(overlap_df.to_string(index=False, columns=["symbol", "date", "score", "setup_class"]))
    else:
        print("  No symbols triggered both Swing and BTST signals on the same day.")
    print("=" * 80 + "\n")
    return 0

if __name__ == "__main__":
    main()