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
from screener import evaluate, _add_indicators
from btst_screener import evaluate_btst

def run_last_week_evaluation():
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
    
    # Last week trading dates in DB (July 15, 2026 to July 22, 2026)
    eval_start = date(2026, 7, 15)
    eval_end = date(2026, 7, 22)
    
    eval_dates = [d.date() for d in sample_df.index if eval_start <= d.date() <= eval_end]
    eval_dates.sort()

    swing_signals = []
    btst_signals = []

    for eval_d in eval_dates:
        d_ts = pd.Timestamp(eval_d)
        daily_swing = []
        
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
                
            # --- SWING EVALUATION ---
            for sc in ["pullback", "trend"]:
                try:
                    cand = evaluate(sym, hist_slice, stock_info={}, skip_fundamental=True, setup_class=sc, index_df=index_df)
                    if cand and (cand.win_rate >= 80.0 or (cand.score >= 5.0 and cand.adx >= 22)):
                        daily_swing.append((cand.win_rate if cand.win_rate > 0 else (cand.score * 15), cand.beta, cand.adx, sym, cand, future_slice, sc))
                except Exception:
                    pass

            # --- BTST EVALUATION ---
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
                    
                    exit_p = next_close
                    reason = "next_day_close"
                    if next_open >= target_p:
                        exit_p = next_open
                        reason = "gap_up_target"
                    elif next_open <= sl_p:
                        exit_p = next_open
                        reason = "gap_down_stop"
                    elif next_high >= target_p:
                        exit_p = target_p
                        reason = "target_hit"
                    elif next_low <= sl_p:
                        exit_p = sl_p
                        reason = "stop_loss_hit"
                        
                    raw_ret = (exit_p - entry_p) / entry_p * 100
                    net_ret = raw_ret - 0.50
                    
                    btst_signals.append({
                        "signal_date": eval_d,
                        "exit_date": future_slice.index[0].date(),
                        "symbol": sym,
                        "entry_price": round(entry_p, 2),
                        "exit_price": round(exit_p, 2),
                        "net_return_pct": round(net_ret, 2),
                        "outcome": "WIN" if net_ret > 0 else "LOSS",
                        "exit_reason": reason,
                        "win_rate_tag": btst_cand.win_rate
                    })
            except Exception:
                pass

        # Sort Swing candidates and select top 5 per day
        daily_swing.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        top_5_swing = daily_swing[:5]

        for conv, beta, adx, sym, cand, future_slice, sc in top_5_swing:
            entry_p = float(future_slice.iloc[0]["Open"])
            target_p = cand.target
            sl_p = cand.stop_loss
            
            outcome = "open_in_progress"
            exit_p = float(future_slice.iloc[-1]["Close"])
            days_held = 0
            
            for f_idx in range(len(future_slice)):
                days_held = f_idx + 1
                row_f = future_slice.iloc[f_idx]
                f_high = float(row_f["High"])
                f_low = float(row_f["Low"])
                
                if f_low <= sl_p and f_high >= target_p:
                    outcome = "stop_loss_hit"
                    exit_p = sl_p
                    break
                elif f_low <= sl_p:
                    outcome = "stop_loss_hit"
                    exit_p = sl_p
                    break
                elif f_high >= target_p:
                    outcome = "target_hit"
                    exit_p = target_p
                    break
                    
            net_ret = (exit_p - entry_p) / entry_p * 100 - 0.50
            swing_signals.append({
                "signal_date": eval_d,
                "symbol": sym,
                "setup_class": sc,
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_p, 2),
                "stop_loss": round(sl_p, 2),
                "target": round(target_p, 2),
                "net_return_pct": round(net_ret, 2),
                "outcome": "WIN" if net_ret > 0 else "LOSS",
                "exit_reason": outcome,
                "win_rate_tag": cand.win_rate,
                "days_held": days_held
            })

    df_swing = pd.DataFrame(swing_signals)
    df_btst = pd.DataFrame(btst_signals)

    print("\n" + "=" * 95)
    print(f"             LAST WEEK PERFORMANCE REPORT ({eval_start} to {eval_end})")
    print("=" * 95)

    print("\n📊 [SWING TRADES REPORT]")
    if not df_swing.empty:
        print(df_swing.to_string(index=False))
        s_wins = (df_swing["outcome"] == "WIN").sum()
        s_tot = len(df_swing)
        s_wr = s_wins / s_tot * 100
        s_avg = df_swing["net_return_pct"].mean()
        s_cum = df_swing["net_return_pct"].sum()
        print("-" * 95)
        print(f"  Swing Total Trades         : {s_tot}")
        print(f"  Swing Winning Trades       : {s_wins}")
        print(f"  Swing Accuracy (Win Rate %): {s_wr:.2f}%")
        print(f"  Swing Average Net Return   : {s_avg:+.2f}% / trade")
        print(f"  Swing Cumulative Return    : {s_cum:+.2f}%")
    else:
        print("No swing signals triggered in last week.")

    print("\n⚡ [BTST TRADES REPORT]")
    if not df_btst.empty:
        print(df_btst.to_string(index=False))
        b_wins = (df_btst["outcome"] == "WIN").sum()
        b_tot = len(df_btst)
        b_wr = b_wins / b_tot * 100
        b_avg = df_btst["net_return_pct"].mean()
        b_cum = df_btst["net_return_pct"].sum()
        print("-" * 95)
        print(f"  BTST Total Trades          : {b_tot}")
        print(f"  BTST Winning Trades        : {b_wins}")
        print(f"  BTST Accuracy (Win Rate % ): {b_wr:.2f}%")
        print(f"  BTST Average Net Return    : {b_avg:+.2f}% / trade")
        print(f"  BTST Cumulative Return     : {b_cum:+.2f}%")
    else:
        print("No BTST signals triggered in last week.")

    print("\n💰 [COMBINED OVERALL LAST WEEK RESULTS]")
    tot_comb = (len(df_swing) if not df_swing.empty else 0) + (len(df_btst) if not df_btst.empty else 0)
    wins_comb = ((df_swing["outcome"] == "WIN").sum() if not df_swing.empty else 0) + ((df_btst["outcome"] == "WIN").sum() if not df_btst.empty else 0)
    wr_comb = wins_comb / tot_comb * 100 if tot_comb > 0 else 0.0
    cum_comb = ((df_swing["net_return_pct"].sum()) if not df_swing.empty else 0.0) + ((df_btst["net_return_pct"].sum()) if not df_btst.empty else 0.0)
    avg_comb = cum_comb / tot_comb if tot_comb > 0 else 0.0

    print(f"  Total Combined Trades      : {tot_comb}")
    print(f"  Total Winning Trades       : {wins_comb}")
    print(f"  Overall Combined Win Rate  : {wr_comb:.2f}%")
    print(f"  Average Return per Position: {avg_comb:+.2f}%")
    print(f"  Combined Net Weekly Return : {cum_comb:+.2f}%")
    print("=" * 95 + "\n")
    return 0

if __name__ == "__main__":
    run_last_week_evaluation()
