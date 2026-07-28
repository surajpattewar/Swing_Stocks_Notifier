#!/usr/bin/env python3
import os
import sys
import duckdb
import numpy as np
import pandas as pd
import ta
from datetime import date, timedelta

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config import config
from screener import evaluate, _add_indicators
from btst_screener import evaluate_btst

def run_1month_evaluation():
    db_path = config.DUCKDB_PATH
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} does not exist.")
        return 1

    con = duckdb.connect(db_path, read_only=True)
    
    # 1 Month evaluation dates (June 1, 2026 to July 1, 2026)
    eval_start = date(2026, 6, 1)
    eval_end = date(2026, 7, 1)
    
    print(f"\n" + "=" * 80)
    print(f"EVALUATING 1-MONTH PERFORMANCE ({eval_start} to {eval_end})")
    print(f"Constraints: Top 5 High Conviction Swing Trades / Day | BTST Signals")
    print("=" * 80 + "\n")

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
                
            # --- SWING EVALUATION ---
            for sc in ["pullback", "trend"]:
                try:
                    cand = evaluate(sym, hist_slice, stock_info={}, skip_fundamental=True, setup_class=sc, index_df=index_df)
                    # Strict high conviction filter: win rate >= 80% or score >= 5.0 with ADX >= 22
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
                    
                    if next_open >= target_p or next_open <= sl_p:
                        ret_pct = (next_open - entry_p) / entry_p * 100
                    elif next_high >= target_p:
                        ret_pct = 2.0
                    elif next_low <= sl_p:
                        ret_pct = -1.2
                    else:
                        ret_pct = (next_close - entry_p) / entry_p * 100
                        
                    net_ret = ret_pct - 0.50 # round-trip fee drag
                    
                    btst_signals.append({
                        "date": eval_d,
                        "symbol": sym,
                        "entry": entry_p,
                        "net_return": net_ret,
                        "is_win": net_ret > 0,
                        "win_rate_tag": btst_cand.win_rate
                    })
            except Exception:
                pass

        # Sort Swing candidates by conviction score and select top 5
        daily_swing.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        top_5_swing = daily_swing[:5]

        for conv, beta, adx, sym, cand, future_slice, sc in top_5_swing:
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
                    
            ret_pct = (exit_p - entry_p) / entry_p * 100 - 0.50
            swing_signals.append({
                "date": eval_d,
                "symbol": sym,
                "score": cand.score,
                "setup_class": sc,
                "outcome": outcome,
                "net_return": ret_pct,
                "is_win": ret_pct > 0,
                "days_held": days_held,
                "win_rate_tag": cand.win_rate
            })

    # --- COMPUTE METRICS ---
    swing_df = pd.DataFrame(swing_signals)
    btst_df = pd.DataFrame(btst_signals)

    print("=" * 80)
    print("                    1-MONTH PERFORMANCE REPORT")
    print("=" * 80)
    
    # 1. SWING METRICS
    print("\n📊 [SWING MODEL PERFORMANCE — HIGH CONVICTION TRADES]")
    if not swing_df.empty:
        total_swing = len(swing_df)
        wins_swing = swing_df["is_win"].sum()
        wr_swing = wins_swing / total_swing * 100
        sum_ret_swing = swing_df["net_return"].sum()
        avg_ret_swing = swing_df["net_return"].mean()
        
        gross_win_s = swing_df.loc[swing_df["net_return"] > 0, "net_return"].sum()
        gross_loss_s = -swing_df.loc[swing_df["net_return"] < 0, "net_return"].sum()
        pf_swing = gross_win_s / gross_loss_s if gross_loss_s > 0 else 99.9
        
        target_hits = (swing_df["outcome"] == "target_hit").sum()
        stop_hits = (swing_df["outcome"] == "stop_loss_hit").sum()
        timeouts = (swing_df["outcome"] == "timeout").sum()

        print(f"  Total Trades Triggered      : {total_swing} (~{round(total_swing / len(eval_dates), 1)} trades/day)")
        print(f"  Total Winning Trades        : {wins_swing}")
        print(f"  Model Accuracy (Win Rate %) : {wr_swing:.2f}%")
        print(f"  Target Hit Ratio            : {target_hits / total_swing * 100:.1f}% ({target_hits} trades)")
        print(f"  Stop Loss Hit Ratio         : {stop_hits / total_swing * 100:.1f}% ({stop_hits} trades)")
        print(f"  Timeout Exits (15 days)     : {timeouts / total_swing * 100:.1f}% ({timeouts} trades)")
        print(f"  Average Return per Trade    : {avg_ret_swing:+.2f}%")
        print(f"  Cumulative 1-Month Return   : {sum_ret_swing:+.2f}%")
        print(f"  Profit Factor               : {pf_swing:.2f}")
    else:
        print("  No swing trades triggered in the last 1 month.")

    # 2. BTST METRICS
    print("\n⚡ [BTST MODEL PERFORMANCE — ALL QUALIFYING TRADES]")
    if not btst_df.empty:
        total_btst = len(btst_df)
        wins_btst = btst_df["is_win"].sum()
        wr_btst = wins_btst / total_btst * 100
        sum_ret_btst = btst_df["net_return"].sum()
        avg_ret_btst = btst_df["net_return"].mean()
        
        gross_win_b = btst_df.loc[btst_df["net_return"] > 0, "net_return"].sum()
        gross_loss_b = -btst_df.loc[btst_df["net_return"] < 0, "net_return"].sum()
        pf_btst = gross_win_b / gross_loss_b if gross_loss_b > 0 else 99.9

        print(f"  Total Trades Triggered      : {total_btst} (~{round(total_btst / len(eval_dates), 1)} trades/day)")
        print(f"  Total Winning Trades        : {wins_btst}")
        print(f"  Model Accuracy (Win Rate %) : {wr_btst:.2f}%")
        print(f"  Average Return per Trade    : {avg_ret_btst:+.2f}%")
        print(f"  Cumulative 1-Month Return   : {sum_ret_btst:+.2f}%")
        print(f"  Profit Factor               : {pf_btst:.2f}")
    else:
        print("  No BTST trades triggered in the last 1 month.")

    # 3. COMBINED SUMMARY
    print("\n💰 [COMBINED 1-MONTH SUMMARY]")
    tot_trades_all = len(swing_df) + len(btst_df)
    tot_wins_all = (swing_df["is_win"].sum() if not swing_df.empty else 0) + (btst_df["is_win"].sum() if not btst_df.empty else 0)
    comb_wr = tot_wins_all / tot_trades_all * 100 if tot_trades_all > 0 else 0.0
    comb_sum_ret = (swing_df["net_return"].sum() if not swing_df.empty else 0.0) + (btst_df["net_return"].sum() if not btst_df.empty else 0.0)
    
    print(f"  Total Combined Trades       : {tot_trades_all}")
    print(f"  Overall Combined Win Rate   : {comb_wr:.2f}%")
    print(f"  Combined Net Return %       : {comb_sum_ret:+.2f}%")
    print("=" * 80 + "\n")

    return 0

if __name__ == "__main__":
    run_1month_evaluation()
