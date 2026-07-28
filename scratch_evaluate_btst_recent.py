#!/usr/bin/env python3
import os
import sys
import duckdb
import numpy as np
import pandas as pd
import ta
from datetime import date, timedelta

_parent_dir = os.path.abspath(os.path.dirname(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config import config
from screener import _add_indicators
from btst_screener import evaluate_btst, CUSTOM_WEIGHTS

def run_recent_btst_evaluation():
    db_path = config.DUCKDB_PATH
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} does not exist.")
        return 1

    con = duckdb.connect(db_path, read_only=True)
    
    # Load Nifty index benchmark
    index_raw = con.execute(
        """
        SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
               close AS Close
        FROM stock_prices
        WHERE symbol = 'NSEI'
        ORDER BY date
        """
    ).fetchdf()
    index_raw["Date"] = pd.to_datetime(index_raw["Date"])
    index_df = index_raw.set_index("Date")

    # Load pricing data
    raw_prices = con.execute(
        """
        SELECT symbol, CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
               open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume,
               delivery_pct AS delivery_pct
        FROM stock_prices
        WHERE symbol != 'NSEI'
        ORDER BY symbol, date
        """
    ).fetchdf()
    con.close()

    symbols_map = {}
    for sym, df_group in raw_prices.groupby("symbol"):
        if len(df_group) < 60:
            continue
        df_group["Date"] = pd.to_datetime(df_group["Date"])
        df_stock = df_group.set_index("Date")
        try:
            symbols_map[sym] = _add_indicators(df_stock)
        except Exception:
            pass

    sample_df = list(symbols_map.values())[0]
    
    # Evaluation dates in July 2026 up to latest available (July 22, 2026)
    eval_start = date(2026, 7, 1)
    eval_end = date(2026, 7, 22)
    
    eval_dates = [d.date() for d in sample_df.index if eval_start <= d.date() <= eval_end]
    eval_dates.sort()

    btst_signals = []

    for eval_d in eval_dates:
        d_ts = pd.Timestamp(eval_d)
        
        for sym, df_stock in symbols_map.items():
            if d_ts not in df_stock.index:
                continue
                
            idx_pos = df_stock.index.get_loc(d_ts)
            if idx_pos < 60:
                continue
                
            hist_slice = df_stock.iloc[:idx_pos + 1].copy()
            future_slice = df_stock.iloc[idx_pos + 1:].copy()
            if future_slice.empty:
                continue
                
            try:
                btst_cand = evaluate_btst(sym, hist_slice, index_df=index_df, skip_event_risk=True)
                if btst_cand and (btst_cand.win_rate >= 80.0 or "Matched custom stock-specific parameters" in btst_cand.reasons):
                    entry_p = float(hist_slice.iloc[-1]["Close"])
                    next_open = float(future_slice.iloc[0]["Open"])
                    next_high = float(future_slice.iloc[0]["High"])
                    next_low = float(future_slice.iloc[0]["Low"])
                    next_close = float(future_slice.iloc[0]["Close"])
                    
                    target_p = entry_p * 1.020
                    sl_p = entry_p * 0.988
                    
                    exit_price = next_close
                    reason = "next_day_close"
                    
                    if next_open >= target_p:
                        exit_price = next_open
                        reason = "gap_up_target"
                    elif next_open <= sl_p:
                        exit_price = next_open
                        reason = "gap_down_stop"
                    elif next_high >= target_p:
                        exit_price = target_p
                        reason = "target_hit"
                    elif next_low <= sl_p:
                        exit_price = sl_p
                        reason = "stop_loss_hit"
                        
                    raw_ret = (exit_price - entry_p) / entry_p * 100
                    net_ret = raw_ret - 0.50 # 0.50% round trip drag
                    
                    btst_signals.append({
                        "signal_date": eval_d,
                        "exit_date": future_slice.index[0].date(),
                        "symbol": sym,
                        "entry_price": round(entry_p, 2),
                        "exit_price": round(exit_price, 2),
                        "raw_return_pct": round(raw_ret, 2),
                        "net_return_pct": round(net_ret, 2),
                        "outcome": "WIN" if net_ret > 0 else "LOSS",
                        "exit_reason": reason,
                        "win_rate_tag": btst_cand.win_rate
                    })
            except Exception:
                pass

    df_res = pd.DataFrame(btst_signals)

    print("\n" + "=" * 90)
    print(f"               RECENT BTST SIGNALS PERFORMANCE REPORT (JULY 2026)")
    print("=" * 90)
    
    if not df_res.empty:
        print(df_res.to_string(index=False))
        print("-" * 90)
        
        total_trades = len(df_res)
        wins = (df_res["outcome"] == "WIN").sum()
        losses = total_trades - wins
        wr = wins / total_trades * 100
        avg_ret = df_res["net_return_pct"].mean()
        cum_ret = df_res["net_return_pct"].sum()
        
        print(f"\nSummary Metrics:")
        print(f"  Total Trades Triggered      : {total_trades}")
        print(f"  Winning Trades              : {wins}")
        print(f"  Losing Trades               : {losses}")
        print(f"  Model Accuracy (Win Rate %) : {wr:.2f}%")
        print(f"  Average Net Return / Trade  : {avg_ret:+.2f}%")
        print(f"  Cumulative BTST Net Return  : {cum_ret:+.2f}%")
    else:
        print("No qualifying BTST signals found in the specified window.")
        
    print("=" * 90 + "\n")
    return 0

if __name__ == "__main__":
    run_recent_btst_evaluation()
