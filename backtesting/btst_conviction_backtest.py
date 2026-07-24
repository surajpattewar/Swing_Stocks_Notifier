
# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
import sys
import os
import pandas as pd
import duckdb
from datetime import datetime, timedelta

# Add workspace to sys.path
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from config import config
from btst_screener import evaluate_btst, CUSTOM_WEIGHTS
from backtest import get_local_symbols

def simulate_btst_open_exit(df, signal_date, entry_price):
    future = df.loc[df.index > signal_date].iloc[:1] # BTST is held for exactly 1 day
    if future.empty:
        return {"outcome": "no_data", "return_pct": 0.0}

    row = future.iloc[0]
    round_trip = 0.50  # 50 bps round-trip transaction cost
    
    # Pure BTST: Enter at Close of signal_date, Exit at Open of next trading day
    next_open = float(row["Open"])
    raw_ret = (next_open - entry_price) / entry_price * 100
    net_ret = round(raw_ret - round_trip, 2)
    
    outcome = "win" if net_ret > 0 else "loss"
    return {"outcome": outcome, "return_pct": net_ret}

def main():
    db_path = config.DUCKDB_PATH
    con = duckdb.connect(db_path, read_only=True)
    symbols = get_local_symbols(db_path, max_stocks=180)
    symbols = [s for s in symbols if s != 'NSEI']
    
    # Load Nifty index
    index_raw = con.execute("SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date, close AS Close FROM stock_prices WHERE symbol = 'NSEI' ORDER BY date").fetchdf()
    index_df = pd.DataFrame()
    if not index_raw.empty:
        index_raw["Date"] = pd.to_datetime(index_raw["Date"])
        index_df = index_raw.set_index("Date")
        
    end_date = pd.Timestamp("2026-07-22").date()
    start_date = end_date - timedelta(days=30)
    
    print(f"Loading price history for {len(symbols)} symbols...")
    stock_data = {}
    for symbol in symbols:
        df_raw = con.execute(
            """
            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                   open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume,
                   delivery_pct
            FROM stock_prices
            WHERE symbol = ? AND CAST(timezone('Asia/Kolkata', date) AS DATE) <= ?
            ORDER BY date
            """,
            [symbol, end_date]
        ).fetchdf()
        if not df_raw.empty and len(df_raw) >= 100:
            df_raw["Date"] = pd.to_datetime(df_raw["Date"])
            stock_data[symbol] = df_raw.set_index("Date")
            
    con.close()
    
    if not stock_data:
        print("No stock data loaded.")
        return
        
    sample_df = list(stock_data.values())[0]
    trading_dates = sample_df.index[(sample_df.index.date >= start_date) & (sample_df.index.date <= end_date)].tolist()
    trading_dates = sorted(trading_dates)
    
    # We will collect all triggered trades and associate them with their custom conviction (win rate)
    all_trades = []
    
    print("Evaluating BTST signals...")
    for idx in range(len(trading_dates) - 1):
        dt = trading_dates[idx]
        
        for symbol, df in stock_data.items():
            df_slice = df.loc[df.index <= dt]
            if len(df_slice) < 50:
                continue
                
            try:
                cand = evaluate_btst(symbol, df_slice, index_df=index_df)
                if cand is not None:
                    # Get conviction (training win rate)
                    cfg = CUSTOM_WEIGHTS.get(symbol, {})
                    conviction = cfg.get("win_rate", 0.0)
                    
                    # Entry is the close price of the signal day
                    entry_close = float(df_slice.iloc[-1]["Close"])
                    
                    outcome = simulate_btst_open_exit(df, dt, entry_close)
                    all_trades.append({
                        "symbol": symbol,
                        "date": dt.date(),
                        "conviction": conviction,
                        "outcome": outcome["outcome"],
                        "return_pct": outcome["return_pct"]
                    })
            except Exception:
                continue
                
    if not all_trades:
        print("No BTST trades triggered in the last month.")
        return
        
    trades_df = pd.DataFrame(all_trades)
    
    # Evaluate for conviction thresholds
    thresholds = [70.0, 80.0, 90.0]
    
    print("\n============================================================")
    print("    PURE BTST OPEN-EXIT PERFORMANCE (LAST 30 DAYS)")
    print("============================================================")
    
    for t in thresholds:
        filtered = trades_df[trades_df["conviction"] >= t]
        
        if filtered.empty:
            print(f"\nConviction >= {t}%: No trades triggered.")
            continue
            
        total_trades = len(filtered)
        wins = (filtered["outcome"] == "win").sum()
        losses = (filtered["outcome"] == "loss").sum()
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_ret = filtered["return_pct"].mean()
        tot_ret = filtered["return_pct"].sum()
        
        print(f"\nConviction >= {t}%:")
        print(f"  • Total Trades Triggered: {total_trades}")
        print(f"    - Won (Net > 0)       : {wins}")
        print(f"    - Lost (Net <= 0)     : {losses}")
        print(f"  • Net Win Rate          : {win_rate:.2f}%")
        print(f"  • Average Return / Trade: {avg_ret:+.2f}% (Net of Costs)")
        print(f"  • Total Cumulative Return: {tot_ret:+.2f}%")
        
    print("============================================================")

if __name__ == "__main__":
    main()
