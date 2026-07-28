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
from btst_screener import evaluate_btst

def run_top_gainers_btst_evaluation():
    db_path = config.DUCKDB_PATH
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} does not exist.")
        return 1

    con = duckdb.connect(db_path, read_only=True)
    
    # 2-Month evaluation window (May 22, 2026 to July 22, 2026)
    eval_start = date(2026, 5, 22)
    eval_end = date(2026, 7, 22)
    
    print("\n" + "=" * 95)
    print(f"       BTST TOP-GAINER RECALL & ACCURACY REPORT (LAST 2 MONTHS: {eval_start} to {eval_end})")
    print("=" * 95 + "\n")

    # Load Nifty index benchmark data
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
            [eval_start - timedelta(days=100), eval_end + timedelta(days=30)]
        ).fetchdf()
        if not index_raw.empty:
            index_raw["Date"] = pd.to_datetime(index_raw["Date"])
            index_df = index_raw.set_index("Date")
    except Exception as e:
        print(f"Warning loading index: {e}")

    # Load stock prices from DB
    raw_prices = con.execute(
        """
        SELECT symbol, CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
               open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume,
               delivery_pct AS delivery_pct
        FROM stock_prices
        WHERE symbol != 'NSEI' AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
        ORDER BY symbol, date
        """,
        [eval_start - timedelta(days=700), eval_end + timedelta(days=30)]
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
    eval_dates = [d.date() for d in sample_df.index if eval_start <= d.date() <= eval_end]
    eval_dates.sort()

    btst_signals = []
    top_gainer_recalls = []

    for idx, eval_d in enumerate(eval_dates[:-1]):
        next_d = eval_dates[idx + 1]
        d_ts = pd.Timestamp(eval_d)
        next_ts = pd.Timestamp(next_d)
        
        actual_gainers_day_T = []
        btst_candidates_day_T_minus_1 = set()
        daily_btst_candidates = []
        
        for sym, df_stock in symbols_map.items():
            if d_ts not in df_stock.index or next_ts not in df_stock.index:
                continue
                
            idx_pos = df_stock.index.get_loc(d_ts)
            if idx_pos < 60:
                continue
                
            hist_slice = df_stock.iloc[:idx_pos + 1].copy()
            future_slice = df_stock.iloc[idx_pos + 1:].copy()
            if future_slice.empty:
                continue
                
            prev_close = float(hist_slice.iloc[-1]["Close"])
            day_t_close = float(future_slice.iloc[0]["Close"])
            day_t_gain_pct = (day_t_close - prev_close) / prev_close * 100
            
            if day_t_gain_pct >= 2.5:
                actual_gainers_day_T.append((day_t_gain_pct, sym))

            # Evaluate BTST using trained custom stock weights
            try:
                btst_cand = evaluate_btst(sym, hist_slice, index_df=index_df, skip_event_risk=True)
                if btst_cand and (btst_cand.win_rate >= 80.0 or "Matched custom stock-specific parameters" in btst_cand.reasons):
                    row = hist_slice.iloc[-1]
                    day_range = row["High"] - row["Low"]
                    pcsi = ((row["Close"] - row["Low"]) / day_range) if day_range > 0 else 0.5
                    vol_r = row["Volume"] / row["vol_avg20"] if row["vol_avg20"] > 0 else 1.0
                    today_ret = (row["Close"] - row["Open"]) / row["Open"] * 100
                    
                    score = (btst_cand.win_rate * 2.0) + (pcsi * 20.0) + (min(vol_r, 3.0) * 10.0)
                    daily_btst_candidates.append((score, sym, hist_slice, future_slice, btst_cand))
            except Exception:
                pass

        # Select Top 3 BTST candidates for Day T-1 post-market
        daily_btst_candidates.sort(key=lambda x: x[0], reverse=True)
        top_btst = daily_btst_candidates[:3]
        
        for sc, sym, hist_slice, future_slice, btst_cand in top_btst:
            btst_candidates_day_T_minus_1.add(sym)
            
            # Execution on Day T Market Open
            entry_p = float(future_slice.iloc[0]["Open"])
            target_p = entry_p * 1.020
            sl_p = entry_p * 0.988
            
            next_high = float(future_slice.iloc[0]["High"])
            next_low = float(future_slice.iloc[0]["Low"])
            next_close = float(future_slice.iloc[0]["Close"])
            
            exit_p = next_close
            reason = "next_day_close"
            if float(future_slice.iloc[0]["Open"]) >= target_p:
                exit_p = float(future_slice.iloc[0]["Open"])
                reason = "gap_up_target"
            elif float(future_slice.iloc[0]["Open"]) <= sl_p:
                exit_p = float(future_slice.iloc[0]["Open"])
                reason = "gap_down_stop"
            elif next_high >= target_p:
                exit_p = target_p
                reason = "target_hit"
            elif next_low <= sl_p:
                exit_p = sl_p
                reason = "stop_loss_hit"
                
            net_ret = (exit_p - entry_p) / entry_p * 100 - 0.50
            btst_signals.append({
                "screener_date": eval_d,
                "entry_date": next_d,
                "symbol": sym,
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_p, 2),
                "net_return_pct": round(net_ret, 2),
                "outcome": "WIN" if net_ret > 0 else "LOSS",
                "exit_reason": reason,
                "win_rate_tag": btst_cand.win_rate
            })

        # Sort actual gainers to find Top 10 Gainers of Day T
        actual_gainers_day_T.sort(key=lambda x: x[0], reverse=True)
        top_10_gainers = [s for g, s in actual_gainers_day_T[:10]]
        
        if len(top_10_gainers) > 0:
            captured_count = len(set(top_10_gainers).intersection(btst_candidates_day_T_minus_1))
            recall_pct = (captured_count / len(top_10_gainers)) * 100
            top_gainer_recalls.append({
                "date": next_d,
                "top_10_count": len(top_10_gainers),
                "captured_count": captured_count,
                "recall_pct": recall_pct
            })

    df_btst = pd.DataFrame(btst_signals)
    df_recalls = pd.DataFrame(top_gainer_recalls)

    print("=" * 95)
    print("           BTST MODEL TOP-GAINER RECALL & PERFORMANCE SUMMARY")
    print("=" * 95)

    if not df_recalls.empty:
        avg_recall = df_recalls["recall_pct"].mean()
        days_met_target = (df_recalls["recall_pct"] >= 40.0).sum()
        pct_days_met = days_met_target / len(df_recalls) * 100
        
        print("\n🎯 [TOP GAINER RECALL METRICS (Day T Top Gainers vs Day T-1 BTST Screener)]")
        print(f"  Average Daily Top-Gainer Recall % : {avg_recall:.2f}%")
        print(f"  Days Meeting >= 40% Recall Target: {days_met_target} / {len(df_recalls)} trading days ({pct_days_met:.1f}%)")

    if not df_btst.empty:
        total_trades = len(df_btst)
        wins = (df_btst["outcome"] == "WIN").sum()
        wr = wins / total_trades * 100
        avg_ret = df_btst["net_return_pct"].mean()
        cum_ret = df_btst["net_return_pct"].sum()

        print("\n⚡ [BTST MODEL PERFORMANCE METRICS (Next-Day Open Buy)]")
        print(f"  Total BTST Trades Triggered      : {total_trades}")
        print(f"  Winning Trades                  : {wins}")
        print(f"  Model Accuracy (Win Rate %)      : {wr:.2f}%")
        print(f"  Average Net Return per Trade     : {avg_ret:+.2f}%")
        print(f"  Cumulative BTST Net Return      : {cum_ret:+.2f}%")
    else:
        print("\nNo BTST trades generated in evaluation window.")

    print("=" * 95 + "\n")
    return 0

if __name__ == "__main__":
    run_top_gainers_btst_evaluation()
