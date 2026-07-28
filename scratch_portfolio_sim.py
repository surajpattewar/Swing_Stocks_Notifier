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

def run_portfolio_simulation():
    db_path = config.DUCKDB_PATH
    if not os.path.exists(db_path):
        print(f"Error: DB path {db_path} does not exist.")
        return 1

    con = duckdb.connect(db_path, read_only=True)
    
    # 1 Month evaluation dates (June 1, 2026 to July 1, 2026)
    eval_start = date(2026, 6, 1)
    eval_end = date(2026, 7, 1)
    
    # Load Nifty index benchmark
    index_raw = con.execute(
        """
        SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
               close AS Close
        FROM stock_prices
        WHERE symbol = 'NSEI' AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
        ORDER BY date
        """,
        [eval_start - timedelta(days=100), eval_end]
    ).fetchdf()
    index_raw["Date"] = pd.to_datetime(index_raw["Date"])
    index_df = index_raw.set_index("Date")
    index_df["sma20"] = ta.trend.sma_indicator(index_df["Close"], window=20)
    index_df["sma50"] = ta.trend.sma_indicator(index_df["Close"], window=50)

    # Load pricing data
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
        # Pre-compute indicators for maximum simulation speed
        try:
            df_stock = _add_indicators(df_stock)
            symbols_map[sym] = df_stock
        except Exception:
            pass

    sample_df = list(symbols_map.values())[0]
    eval_dates = [d.date() for d in sample_df.index if eval_start <= d.date() <= eval_end]
    eval_dates.sort()

    # Portfolio simulation parameters:
    # Starting Capital = 100,000 INR
    # Base Position size: 10% of portfolio (max 10 open positions)
    starting_capital = 100000.0
    cash = starting_capital
    open_positions = []
    trade_log = []

    for eval_d in eval_dates:
        d_ts = pd.Timestamp(eval_d)
        
        # 1. Update existing open positions
        active_positions = []
        for pos in open_positions:
            sym = pos["symbol"]
            df_s = symbols_map[sym]
            if d_ts in df_s.index:
                row_today = df_s.loc[d_ts]
                t_open = float(row_today["Open"])
                t_high = float(row_today["High"])
                t_low = float(row_today["Low"])
                t_close = float(row_today["Close"])
                pos["days_held"] += 1
                
                is_exit = False
                exit_price = t_close
                reason = "open"
                
                # BTST 1-day exit rule
                if pos["trade_type"] == "BTST":
                    is_exit = True
                    target_p = pos["entry"] * 1.020
                    sl_p = pos["entry"] * 0.988
                    if t_open >= target_p or t_open <= sl_p:
                        exit_price = t_open
                    elif t_high >= target_p:
                        exit_price = target_p
                    elif t_low <= sl_p:
                        exit_price = sl_p
                    else:
                        exit_price = t_close
                    reason = "btst_overnight_exit"
                else:
                    # Swing trade exit rules
                    if t_low <= pos["stop_loss"]:
                        is_exit = True
                        exit_price = pos["stop_loss"]
                        reason = "stop_loss"
                    elif t_high >= pos["target"]:
                        is_exit = True
                        exit_price = pos["target"]
                        reason = "target_hit"
                    elif pos["days_held"] >= 3 and (t_close - pos["entry"]) / pos["entry"] <= -0.005:
                        # Fast recycling for stagnant/losing trades after 3 days
                        is_exit = True
                        exit_price = t_close
                        reason = "early_stagnant_exit"
                    elif pos["days_held"] >= 8:
                        is_exit = True
                        exit_price = t_close
                        reason = "max_hold_timeout"
                    
                if is_exit:
                    trade_return_pct = (exit_price - pos["entry"]) / pos["entry"] * 100 - 0.50 # 0.50% round-trip drag
                    capital_returned = pos["position_value"] * (1.0 + trade_return_pct / 100.0)
                    cash += capital_returned
                    trade_log.append({
                        "entry_date": pos["entry_date"],
                        "exit_date": eval_d,
                        "symbol": sym,
                        "trade_type": pos["trade_type"],
                        "return_pct": trade_return_pct,
                        "profit_inr": capital_returned - pos["position_value"],
                        "reason": reason
                    })
                else:
                    active_positions.append(pos)
            else:
                active_positions.append(pos)
                
        open_positions = active_positions

        # 2. Market Regime Filter Check (Require Nifty Close > SMA20)
        idx_slice = index_df.loc[index_df.index <= d_ts]
        if not idx_slice.empty:
            last_nifty = idx_slice.iloc[-1]
            if last_nifty["Close"] < last_nifty["sma20"]:
                continue # Skip new positions during benchmark market pullbacks

        # 3. Screen for high-conviction candidates
        if len(open_positions) < 10 and cash > 5000:
            daily_candidates = []
            
            for sym, df_s in symbols_map.items():
                if sym in [p["symbol"] for p in open_positions]:
                    continue
                if d_ts in df_s.index:
                    idx_p = df_s.index.get_loc(d_ts)
                    if idx_p >= 60:
                        hist = df_s.iloc[:idx_p + 1]
                        
                        # --- BTST SCREENING ---
                        try:
                            btst_cand = evaluate_btst(sym, hist, index_df=index_df, skip_event_risk=True)
                            if btst_cand and (btst_cand.win_rate >= 80.0 or "Matched custom stock-specific parameters" in btst_cand.reasons):
                                daily_candidates.append({
                                    "symbol": sym,
                                    "trade_type": "BTST",
                                    "conviction": btst_cand.win_rate if btst_cand.win_rate > 0 else 85.0,
                                    "entry": float(hist.iloc[-1]["Close"]),
                                    "target": float(hist.iloc[-1]["Close"]) * 1.020,
                                    "stop_loss": float(hist.iloc[-1]["Close"]) * 0.988,
                                })
                        except Exception:
                            pass

                        # --- SWING SCREENING ("trend" & "pullback") ---
                        for sc in ["pullback", "trend"]:
                            try:
                                cand = evaluate(sym, hist, stock_info={}, skip_fundamental=True, setup_class=sc, index_df=index_df)
                                # Strict conviction: win rate >= 80% or score >= 5.0 with strong ADX
                                if cand and (cand.win_rate >= 80.0 or (cand.score >= 5.0 and cand.adx >= 22)):
                                    daily_candidates.append({
                                        "symbol": sym,
                                        "trade_type": f"Swing_{sc}",
                                        "conviction": cand.win_rate if cand.win_rate > 0 else (cand.score * 15),
                                        "entry": cand.close,
                                        "target": cand.target,
                                        "stop_loss": cand.stop_loss,
                                    })
                            except Exception:
                                pass

            # Sort by conviction score
            daily_candidates.sort(key=lambda c: c["conviction"], reverse=True)
            
            # Execute top available setups with conviction-scaled position sizing
            slots_available = 10 - len(open_positions)
            for cand in daily_candidates[:slots_available]:
                current_portfolio_value = cash + sum(p["position_value"] for p in open_positions)
                
                # Scale position size: 10% base, 15% for >=85% conviction, 20% for >=90% conviction
                conv = cand["conviction"]
                alloc_pct = 0.20 if conv >= 90.0 else (0.15 if conv >= 85.0 else 0.10)
                position_size = min(cash, current_portfolio_value * alloc_pct)
                if position_size < 3000:
                    break
                cash -= position_size
                open_positions.append({
                    "symbol": cand["symbol"],
                    "trade_type": cand["trade_type"],
                    "entry_date": eval_d,
                    "entry": cand["entry"],
                    "target": cand["target"],
                    "stop_loss": cand["stop_loss"],
                    "position_value": position_size,
                    "days_held": 0
                })

    # Resolve remaining open positions forward in time to reach true outcome
    for pos in open_positions:
        sym = pos["symbol"]
        df_s = symbols_map[sym]
        future_df = df_s.loc[df_s.index.date > pos["entry_date"]]
        
        exit_price = float(df_s.iloc[-1]["Close"])
        reason = "timeout"
        
        for f_idx in range(len(future_df)):
            pos["days_held"] += 1
            row_f = future_df.iloc[f_idx]
            f_open = float(row_f["Open"])
            f_high = float(row_f["High"])
            f_low = float(row_f["Low"])
            f_close = float(row_f["Close"])
            
            if pos["trade_type"] == "BTST":
                target_p = pos["entry"] * 1.020
                sl_p = pos["entry"] * 0.988
                if f_open >= target_p or f_open <= sl_p:
                    exit_price = f_open
                elif f_high >= target_p:
                    exit_price = target_p
                elif f_low <= sl_p:
                    exit_price = sl_p
                else:
                    exit_price = f_close
                reason = "btst_overnight_exit"
                break
            else:
                if f_low <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    reason = "stop_loss"
                    break
                elif f_high >= pos["target"]:
                    exit_price = pos["target"]
                    reason = "target_hit"
                    break
                elif pos["days_held"] >= 3 and (f_close - pos["entry"]) / pos["entry"] <= -0.005:
                    exit_price = f_close
                    reason = "early_stagnant_exit"
                    break
                elif pos["days_held"] >= 8:
                    exit_price = f_close
                    reason = "max_hold_timeout"
                    break

        trade_return_pct = (exit_price - pos["entry"]) / pos["entry"] * 100 - 0.50
        cap_ret = pos["position_value"] * (1.0 + trade_return_pct / 100.0)
        cash += cap_ret
        trade_log.append({
            "entry_date": pos["entry_date"],
            "exit_date": eval_end,
            "symbol": sym,
            "trade_type": pos["trade_type"],
            "return_pct": trade_return_pct,
            "profit_inr": cap_ret - pos["position_value"],
            "reason": reason
        })

    total_portfolio_value = cash
    total_net_profit = total_portfolio_value - starting_capital
    portfolio_return_pct = (total_net_profit / starting_capital) * 100

    df_log = pd.DataFrame(trade_log)

    print("\n" + "=" * 80)
    print("           OPTIMIZED PORTFOLIO EQUITY CURVE SIMULATION (1 MONTH)")
    print("=" * 80)
    print(f"Starting Capital               : ₹{starting_capital:,.2f}")
    print(f"Ending Portfolio Equity        : ₹{total_portfolio_value:,.2f}")
    print(f"Total Net Portfolio Return %   : {portfolio_return_pct:+.2f}% (INR {total_net_profit:+,.2f})")
    print(f"Total Executed Trades          : {len(df_log)}")
    if not df_log.empty:
        wins = (df_log["return_pct"] > 0).sum()
        print(f"Winning Trades / Accuracy      : {wins} / {len(df_log)} ({wins/len(df_log)*100:.2f}% Win Rate)")
        print(f"Average Net Return / Position  : {df_log['return_pct'].mean():+.2f}%")
        print("\nTrade Reason Breakdown:")
        print(df_log["reason"].value_counts().to_string())
        print("\nTrade Type Breakdown:")
        print(df_log["trade_type"].value_counts().to_string())
    print("=" * 80 + "\n")
    return 0

if __name__ == "__main__":
    run_portfolio_simulation()
