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
    warmup_days = 100
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

def run_btst_backtest(symbols_data, start_date, end_date, index_df=None):
    trades = []
    
    # Calculate index SMA50
    idx_df = pd.DataFrame()
    if index_df is not None and not index_df.empty:
        idx_df = index_df.copy()
        idx_df["sma50"] = ta.trend.sma_indicator(idx_df["Close"], window=50)
        
    for symbol, df_all in symbols_data.items():
        if df_all is None or len(df_all) < 60:
            continue
            
        df = df_all.copy()
        df["sma20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["vol_avg20"] = df["Volume"].shift(1).rolling(20).mean()
        df["rsi14"] = ta.momentum.rsi(df["Close"], window=14)
        df = df.dropna()
        
        eval_dates = df.index[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
        
        for idx in range(len(df) - 1):
            dt = df.index[idx]
            if dt not in eval_dates:
                continue
                
            row = df.iloc[idx]
            next_row = df.iloc[idx + 1] # Day D+1
            
            # BTST Screener Criteria:
            # 1. Price is above 20 SMA & 50 SMA (uptrend)
            # 2. Closes within 0.7% of the high of the day
            # 3. Volume is at least 2.0x the 20-day average volume
            # 4. Bullish green candle (Close > Open)
            # 5. Today's return is positive (> 1.5%)
            # 6. RSI(14) is in momentum zone (between 55 and 78)
            
            pct_today = (row["Close"] - row["Open"]) / row["Open"] * 100
            near_high = row["Close"] >= 0.993 * row["High"]
            vol_spike = row["Volume"] > 2.0 * row["vol_avg20"]
            uptrend = row["Close"] > row["sma20"] and row["Close"] > row["sma50"]
            rsi_momentum = 55 <= row["rsi14"] <= 78
            
            # Broader Market Index filter
            index_ok = True
            if not idx_df.empty:
                if dt in idx_df.index and dt in idx_df["sma50"].index:
                    index_ok = idx_df.loc[dt, "Close"] > idx_df.loc[dt, "sma50"]
            
            if uptrend and near_high and vol_spike and row["Close"] > row["Open"] and pct_today > 1.5 and rsi_momentum and index_ok:
                entry_price = float(row["Close"])
                next_open = float(next_row["Open"])
                next_high = float(next_row["High"])
                next_low = float(next_row["Low"])
                next_close = float(next_row["Close"])
                
                # Exit Type A: Sell at Next Day's Open (Immediate Morning gap capture)
                ret_open = (next_open - entry_price) / entry_price * 100
                
                # Exit Type B: Limit Target +1.5% or SL -1.5%, else Close
                target_pct = 1.5
                sl_pct = -1.5
                
                target_price = entry_price * (1 + target_pct/100)
                sl_price = entry_price * (1 + sl_pct/100)
                
                if next_open >= target_price:
                    # Gapped up past target, fill at Open
                    ret_limit = (next_open - entry_price) / entry_price * 100
                    outcome = "win"
                elif next_open <= sl_price:
                    # Gapped down past SL, fill at Open
                    ret_limit = (next_open - entry_price) / entry_price * 100
                    outcome = "loss"
                elif next_high >= target_price:
                    ret_limit = target_pct
                    outcome = "win"
                elif next_low <= sl_price:
                    ret_limit = sl_pct
                    outcome = "loss"
                else:
                    # Exit at Close
                    ret_limit = (next_close - entry_price) / entry_price * 100
                    outcome = "win" if ret_limit > 0 else "loss"
                    
                trades.append({
                    "symbol": symbol,
                    "date": dt.date(),
                    "entry": entry_price,
                    "exit_open": next_open,
                    "exit_close": next_close,
                    "ret_open": round(ret_open, 2),
                    "ret_limit": round(ret_limit, 2),
                    "outcome": outcome
                })
                
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
        
    # Load Nifty index data
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
            [start_date - timedelta(days=100), end_date]
        ).fetchdf()
        if not index_raw.empty:
            index_raw["Date"] = pd.to_datetime(index_raw["Date"])
            index_df = index_raw.set_index("Date")
    except Exception as e:
        print(f"Warning: Could not load Nifty index benchmark from DuckDB: {e}")
        
    con.close()
    
    df_trades = run_btst_backtest(symbols_data, start_date, end_date, index_df=index_df)
    
    if df_trades.empty:
        print("No BTST trades generated.")
        return
        
    print("\n" + "=" * 60)
    print("                    BTST BACKTEST REPORT")
    print("=" * 60)
    print(f"Period         : {start_date} to {end_date}")
    print(f"Total Trades   : {len(df_trades)}")
    
    # Analyze Exit Type A: Sell at Next Day's Open
    wins_open = (df_trades["ret_open"] > 0).sum()
    win_rate_open = wins_open / len(df_trades) * 100
    avg_ret_open = df_trades["ret_open"].mean()
    
    print("-" * 60)
    print("EXIT TYPE A: Sell at Next Day's OPEN (Gap Capture)")
    print(f"  Win Rate (Positive Return): {win_rate_open:.2f}%")
    print(f"  Average Return            : {avg_ret_open:+.2f}%")
    
    # Analyze Exit Type B: Limit Target +1.5% / SL -1.5%
    wins_limit = (df_trades["ret_limit"] > 0).sum()
    win_rate_limit = wins_limit / len(df_trades) * 100
    avg_ret_limit = df_trades["ret_limit"].mean()
    
    gross_win = df_trades.loc[df_trades["ret_limit"] > 0, "ret_limit"].sum()
    gross_loss = -df_trades.loc[df_trades["ret_limit"] < 0, "ret_limit"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else 99.9
    
    print("-" * 60)
    print("EXIT TYPE B: Limit Target +1.5% / SL -1.5% (Intraday execution)")
    print(f"  Wins / Losses             : {wins_limit} / {len(df_trades) - wins_limit}")
    print(f"  Win Rate                  : {win_rate_limit:.2f}%")
    print(f"  Average Return            : {avg_ret_limit:+.2f}%")
    print(f"  Profit Factor             : {pf:.2f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
