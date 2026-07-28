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

def run_conviction_evaluation():
    db_path = config.DUCKDB_PATH
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} does not exist.")
        return 1

    con = duckdb.connect(db_path, read_only=True)
    
    # 2-Month evaluation dates (May 22, 2026 to July 22, 2026)
    eval_start = date(2026, 5, 22)
    eval_end = date(2026, 7, 22)
    
    btst_thresh = config.BTST_MIN_CONVICTION_PCT
    swing_thresh = config.SWING_MIN_CONVICTION_PCT
    
    print("\n" + "=" * 95)
    print(f"       CONVICTION EVALUATION REPORT — LAST 2 MONTHS ({eval_start} to {eval_end})")
    print(f"   Filters: BTST Conviction >= {btst_thresh:.1f}% | Swing Conviction >= {swing_thresh:.1f}%")
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
                
            # --- SWING EVALUATION (Strict conviction >= swing_thresh %) ---
            for sc in ["pullback", "trend"]:
                try:
                    cand = evaluate(sym, hist_slice, stock_info={}, skip_fundamental=True, setup_class=sc, index_df=index_df)
                    if cand and cand.win_rate >= swing_thresh:
                        daily_swing.append((cand.win_rate, cand.beta, cand.adx, sym, cand, future_slice, sc))
                except Exception:
                    pass

        # Rank and pick Top 5 Swing trades per day
        daily_swing.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        for conv, beta, adx, sym, cand, future_slice, sc in daily_swing[:5]:
            entry_p = float(future_slice.iloc[0]["Open"])
            target_p = cand.target
            sl_p = cand.stop_loss
            
            outcome = "timeout"
            exit_p = float(future_slice.iloc[min(14, len(future_slice) - 1)]["Close"])
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
                    
                if days_held >= 15:
                    break
                    
            net_ret = (exit_p - entry_p) / entry_p * 100 - 0.50
            swing_signals.append({
                "date": eval_d,
                "symbol": sym,
                "setup_class": sc,
                "entry_price": round(entry_p, 2),
                "exit_price": round(exit_p, 2),
                "net_return_pct": round(net_ret, 2),
                "outcome": "WIN" if net_ret > 0 else "LOSS",
                "exit_reason": outcome,
                "win_rate_tag": cand.win_rate
            })

            # --- BTST EVALUATION (Strict conviction >= btst_thresh %) ---
            try:
                btst_cand = evaluate_btst(sym, hist_slice, index_df=index_df, skip_event_risk=True)
                if btst_cand and btst_cand.win_rate >= btst_thresh:
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
                        "date": eval_d,
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

    df_swing = pd.DataFrame(swing_signals)
    df_btst = pd.DataFrame(btst_signals)

    print("=" * 95)
    print("                2-MONTH CONVICTION PERFORMANCE REPORT")
    print("=" * 95)
    
    # 1. SWING METRICS
    print(f"\n📊 [SWING MODEL PERFORMANCE — CONVICTION >= {swing_thresh:.1f}%]")
    if not df_swing.empty:
        total_swing = len(df_swing)
        wins_swing = (df_swing["outcome"] == "WIN").sum()
        wr_swing = wins_swing / total_swing * 100
        sum_ret_swing = df_swing["net_return_pct"].sum()
        avg_ret_swing = df_swing["net_return_pct"].mean()
        
        gross_win_s = df_swing.loc[df_swing["net_return_pct"] > 0, "net_return_pct"].sum()
        gross_loss_s = -df_swing.loc[df_swing["net_return_pct"] < 0, "net_return_pct"].sum()
        pf_swing = gross_win_s / gross_loss_s if gross_loss_s > 0 else 99.9

        print(f"  Total Swing Trades Triggered: {total_swing} (~{round(total_swing / len(eval_dates), 1)} trades/day)")
        print(f"  Total Winning Trades        : {wins_swing}")
        print(f"  Model Accuracy (Win Rate %) : {wr_swing:.2f}%")
        print(f"  Average Return per Trade    : {avg_ret_swing:+.2f}%")
        print(f"  Cumulative 2-Month Return   : {sum_ret_swing:+.2f}%")
        print(f"  Profit Factor               : {pf_swing:.2f}")
    else:
        print(f"  No Swing trades met the >= {swing_thresh:.1f}% conviction threshold.")

    # 2. BTST METRICS
    print(f"\n⚡ [BTST MODEL PERFORMANCE — CONVICTION >= {btst_thresh:.1f}%]")
    if not df_btst.empty:
        total_btst = len(df_btst)
        wins_btst = (df_btst["outcome"] == "WIN").sum()
        wr_btst = wins_btst / total_btst * 100
        sum_ret_btst = df_btst["net_return_pct"].sum()
        avg_ret_btst = df_btst["net_return_pct"].mean()
        
        gross_win_b = df_btst.loc[df_btst["net_return_pct"] > 0, "net_return_pct"].sum()
        gross_loss_b = -df_btst.loc[df_btst["net_return_pct"] < 0, "net_return_pct"].sum()
        pf_btst = gross_win_b / gross_loss_b if gross_loss_b > 0 else 99.9

        print(f"  Total BTST Trades Triggered : {total_btst} (~{round(total_btst / len(eval_dates), 1)} trades/day)")
        print(f"  Total Winning Trades        : {wins_btst}")
        print(f"  Model Accuracy (Win Rate %) : {wr_btst:.2f}%")
        print(f"  Average Return per Trade    : {avg_ret_btst:+.2f}%")
        print(f"  Cumulative 2-Month Return   : {sum_ret_btst:+.2f}%")
        print(f"  Profit Factor               : {pf_btst:.2f}")
    else:
        print(f"  No BTST trades met the >= {btst_thresh:.1f}% conviction threshold.")

    # 3. COMBINED SUMMARY
    print("\n💰 [COMBINED OVERALL 2-MONTH SUMMARY]")
    tot_trades_all = (len(df_swing) if not df_swing.empty else 0) + (len(df_btst) if not df_btst.empty else 0)
    tot_wins_all = ((df_swing["outcome"] == "WIN").sum() if not df_swing.empty else 0) + ((df_btst["outcome"] == "WIN").sum() if not df_btst.empty else 0)
    comb_wr = tot_wins_all / tot_trades_all * 100 if tot_trades_all > 0 else 0.0
    comb_sum_ret = ((df_swing["net_return_pct"].sum()) if not df_swing.empty else 0.0) + ((df_btst["net_return_pct"].sum()) if not df_btst.empty else 0.0)
    comb_avg_ret = comb_sum_ret / tot_trades_all if tot_trades_all > 0 else 0.0

    print(f"  Total Combined Trades       : {tot_trades_all}")
    print(f"  Total Winning Trades        : {tot_wins_all}")
    print(f"  Overall Combined Accuracy   : {comb_wr:.2f}%")
    print(f"  Average Return per Trade    : {comb_avg_ret:+.2f}%")
    print(f"  Combined 2-Month Return %   : {comb_sum_ret:+.2f}%")
    print("=" * 95 + "\n")

    return 0

if __name__ == "__main__":
    run_conviction_evaluation()
