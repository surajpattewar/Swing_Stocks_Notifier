import os
import duckdb
import pandas as pd
import numpy as np
import ta
from datetime import date, timedelta

DB_PATH = "data/duckdb/screener_data.duckdb"

def get_symbols(con):
    rows = con.execute("SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol").fetchall()
    return [r[0] for r in rows if r[0] != 'NSEI']

def load_data(con, symbol, start_date, end_date):
    warmup_days = 300
    start_warm = start_date - timedelta(days=warmup_days)
    df = con.execute(
        """
        SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
               open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
        FROM stock_prices
        WHERE symbol = ? AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
        ORDER BY date
        """,
        [symbol, start_warm, end_date]
    ).fetchdf()
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")

def load_index(con, start_date, end_date):
    warmup_days = 300
    start_warm = start_date - timedelta(days=warmup_days)
    df = con.execute(
        """
        SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
               close AS Close
        FROM stock_prices
        WHERE symbol = 'NSEI' AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
        ORDER BY date
        """,
        [start_warm, end_date]
    ).fetchdf()
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")

def backtest_rsi2_opt(symbols_data, index_df, start_date, end_date, rsi_entry=10, max_holding_days=15):
    trades = []
    
    # Calculate index indicators
    idx = index_df.copy()
    idx["index_sma200"] = ta.trend.sma_indicator(idx["Close"], window=200)
    idx = idx.dropna()
    
    for symbol, df_all in symbols_data.items():
        if df_all is None or len(df_all) < 220:
            continue
            
        df = df_all.copy()
        df["sma200"] = ta.trend.sma_indicator(df["Close"], window=200)
        df["sma5"] = ta.trend.sma_indicator(df["Close"], window=5)
        df["rsi2"] = ta.momentum.rsi(df["Close"], window=2)
        df = df.dropna()
        
        # Merge index indicators
        df = df.join(idx[["index_sma200"]], how="inner")
        
        eval_dates = df.index[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
        
        in_trade = False
        entry_price = 0.0
        entry_date = None
        days_in_trade = 0
        
        for dt in eval_dates:
            row = df.loc[dt]
            
            if in_trade:
                days_in_trade += 1
                # Exit condition: close above SMA5 or max holding days hit
                if row["Close"] > row["sma5"] or days_in_trade >= max_holding_days:
                    exit_price = float(row["Close"])
                    ret = (exit_price - entry_price) / entry_price * 100
                    trades.append({
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "exit_date": dt.date(),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "days_held": days_in_trade,
                        "return_pct": round(ret, 2),
                        "outcome": "win" if ret > 0 else "loss"
                    })
                    in_trade = False
            else:
                # Entry condition:
                # 1. Stock price > 200 SMA (stock uptrend)
                # 2. Stock RSI(2) < rsi_entry (stock oversold)
                # 3. Index close > index 200 SMA (index uptrend)
                # 4. We check if Nifty index itself is not oversold (e.g. index is healthy)
                
                # Fetch Nifty index close for today
                index_close = idx.loc[dt, "Close"] if dt in idx.index else 0.0
                index_sma200 = idx.loc[dt, "index_sma200"] if dt in idx.index else 0.0
                
                if row["Close"] > row["sma200"] and row["rsi2"] < rsi_entry and index_close > index_sma200:
                    entry_price = float(row["Close"])
                    entry_date = dt.date()
                    in_trade = True
                    days_in_trade = 0
                    
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)

def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    symbols = get_symbols(con)
    latest_date_row = con.execute("SELECT max(date) FROM stock_prices").fetchone()
    end_date = pd.Timestamp(latest_date_row[0]).date()
    start_date = end_date - timedelta(days=365) # Backtest 1 year
    
    print(f"Loading data for {len(symbols)} symbols over 1 year...")
    symbols_data = {}
    for sym in symbols:
        symbols_data[sym] = load_data(con, sym, start_date, end_date)
    index_df = load_index(con, start_date, end_date)
    con.close()
    
    for entry_thresh in [2, 3, 5, 10]:
        print(f"\nBacktesting Larry Connors RSI(2) + Nifty Trend Filter (RSI(2) < {entry_thresh})")
        df_trades = backtest_rsi2_opt(symbols_data, index_df, start_date, end_date, rsi_entry=entry_thresh)
        
        if df_trades.empty:
            print("No trades generated.")
            continue
            
        total_trades = len(df_trades)
        wins = (df_trades["outcome"] == "win").sum()
        losses = (df_trades["outcome"] == "loss").sum()
        win_rate = wins / total_trades * 100
        avg_ret = df_trades["return_pct"].mean()
        
        gross_gain = df_trades.loc[df_trades["return_pct"] > 0, "return_pct"].sum()
        gross_loss = -df_trades.loc[df_trades["return_pct"] < 0, "return_pct"].sum()
        pf = gross_gain / gross_loss if gross_loss > 0 else 99.9
        
        print("-" * 60)
        print(f"Total Trades   : {total_trades}")
        print(f"Wins / Losses  : {wins} / {losses}")
        print(f"Win Rate       : {win_rate:.2f}%")
        print(f"Avg Return     : {avg_ret:+.2f}%")
        print(f"Profit Factor  : {pf:.2f}")
        print(f"Avg Days Held  : {df_trades['days_held'].mean():.1f} days")
        print("-" * 60)

if __name__ == "__main__":
    main()
