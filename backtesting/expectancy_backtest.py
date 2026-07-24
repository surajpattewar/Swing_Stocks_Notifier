
# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
import sys
import os
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, timedelta

# Add the workspace root to sys.path
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from config import config
from backtest import get_local_symbols
from screener import evaluate

def simulate_outcome_uncapped(df, signal_date, entry_price, stop_loss, target, max_holding_days=22):
    future = df.loc[df.index > signal_date].iloc[:max_holding_days]
    if future.empty:
        return {"outcome": "no_data", "return_pct": 0.0}

    round_trip = 0.50  # 50 bps round-trip transaction cost
    for i, (dt, row) in enumerate(future.iterrows(), start=1):
        hit_target = row["High"] >= target
        hit_stop = row["Low"] <= stop_loss
        if hit_stop:
            exit_price = stop_loss
            raw_ret = (exit_price - entry_price) / entry_price * 100
            return {"outcome": "stop_loss_hit", "return_pct": round(raw_ret - round_trip, 2)}
        if hit_target:
            exit_price = target
            raw_ret = (exit_price - entry_price) / entry_price * 100
            return {"outcome": "target_hit", "return_pct": round(raw_ret - round_trip, 2)}

    last_row = future.iloc[-1]
    exit_price = float(last_row["Close"])
    raw_ret = (exit_price - entry_price) / entry_price * 100
    return {"outcome": "timeout", "return_pct": round(raw_ret - round_trip, 2)}

def main():
    db_path = config.DUCKDB_PATH
    con = duckdb.connect(db_path, read_only=True)
    symbols = get_local_symbols(db_path, max_stocks=100)
    symbols = [s for s in symbols if s != 'NSEI']
    
    # Load Nifty index
    index_raw = con.execute("SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date, close AS Close FROM stock_prices WHERE symbol = 'NSEI' ORDER BY date").fetchdf()
    index_raw["Date"] = pd.to_datetime(index_raw["Date"])
    index_df = index_raw.set_index("Date")
    
    backtest_end = pd.Timestamp("2026-07-22").date()
    backtest_start = backtest_end - timedelta(days=45)
    
    all_trades = []
    
    print(f"Running expectancy-optimized backtest with custom weights from {backtest_start} to {backtest_end}...")
    for symbol in symbols:
        df_raw = con.execute(
            "SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume FROM stock_prices WHERE symbol = ? ORDER BY date",
            [symbol]
        ).fetchdf()
        if df_raw.empty or len(df_raw) < 150:
            continue
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df = df_raw.set_index("Date")
        
        # Filter for dates in backtest range
        eval_dates = df.index[(df.index >= pd.Timestamp(backtest_start)) & (df.index <= pd.Timestamp(backtest_end))]
        
        for dt in eval_dates:
            df_slice = df.loc[df.index <= dt]
            if len(df_slice) < 50:
                continue
                
            for sc in ["trend", "pullback"]:
                try:
                    cand = evaluate(symbol, df_slice, skip_fundamental=True, index_df=index_df, use_custom_weights=True, setup_class=sc)
                    
                    # Threshold for screening
                    if cand.score >= 5.0:
                        future_bars = df.loc[df.index > dt]
                        if future_bars.empty:
                            continue
                        next_open = float(future_bars.iloc[0]["Open"])
                        
                        trade = simulate_outcome_uncapped(df, dt, next_open, cand.stop_loss, cand.target)
                        all_trades.append(trade)
                except Exception:
                    continue
                
    con.close()
    
    if not all_trades:
        print("No trades generated.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    total_trades = len(trades_df)
    decided = trades_df[trades_df["outcome"].isin(["target_hit", "stop_loss_hit"])]
    wins = (decided["outcome"] == "target_hit").sum()
    losses = (decided["outcome"] == "stop_loss_hit").sum()
    win_rate = (wins / len(decided) * 100) if len(decided) > 0 else 0.0
    avg_ret = trades_df["return_pct"].mean()
    
    # Calculate average return of wins and losses
    win_trades = trades_df[trades_df["return_pct"] > 0]
    loss_trades = trades_df[trades_df["return_pct"] < 0]
    avg_win = win_trades["return_pct"].mean() if not win_trades.empty else 0.0
    avg_loss = loss_trades["return_pct"].mean() if not loss_trades.empty else 0.0
    
    print("\n==============================================")
    print("      EXPECTANCY-OPTIMIZED BACKTEST RESULTS")
    print("==============================================")
    print(f"Total Trades Triggered: {total_trades}")
    print(f"Decided (T/SL hits)   : {len(decided)}")
    print(f"  - Target Hits (Wins): {wins}")
    print(f"  - Stop Loss (Losses): {losses}")
    print(f"Decided Win Rate      : {win_rate:.2f}%")
    print(f"Average Return / Trade: {avg_ret:+.2f}%  (Net of Costs!)")
    print(f"Average Profit / Win  : {avg_win:+.2f}%")
    print(f"Average Loss / Loss   : {avg_loss:+.2f}%")
    print("==============================================")

if __name__ == "__main__":
    main()
