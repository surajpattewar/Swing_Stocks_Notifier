
# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
import os
import sys
import duckdb
import pandas as pd
import numpy as np

# Add parent directory to path to allow importing btst_screener

from btst_screener import evaluate_btst
from stock_universe import get_stock_universe

def main():
    db_path = "data/duckdb/screener_data.duckdb"
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return
        
    con = duckdb.connect(db_path, read_only=True)
    
    # Get latest date in the database
    latest_date_row = con.execute("SELECT max(date) FROM stock_prices").fetchone()
    if not latest_date_row or latest_date_row[0] is None:
        print("Error: No data in stock_prices table.")
        con.close()
        return
        
    latest_date = pd.Timestamp(latest_date_row[0]).date()
    print(f"Analyzing BTST Misses from 2026-06-01 to {latest_date}...")
    
    # Find all trading dates in June & July 2026
    trading_dates_df = con.execute("""
        SELECT DISTINCT CAST(timezone('Asia/Kolkata', date) AS DATE) as date_val
        FROM stock_prices
        WHERE date >= '2026-06-01' AND date <= ?
        ORDER BY date_val
    """, [latest_date]).fetchdf()
    trading_dates = [d.strftime('%Y-%m-%d') for d in trading_dates_df['date_val']]
    
    # Load prices to calculate daily returns and find top gainers
    prices_df = con.execute("""
        SELECT symbol, date, open, high, low, close, volume
        FROM stock_prices
        WHERE date >= '2026-05-01' AND date <= ?
        ORDER BY symbol, date
    """, [latest_date]).fetchdf()
    
    prices_df['date_str'] = pd.to_datetime(prices_df['date']).dt.strftime('%Y-%m-%d')
    prices_df['prev_close'] = prices_df.groupby('symbol')['close'].shift(1)
    prices_df['return_pct'] = (prices_df['close'] - prices_df['prev_close']) / prices_df['prev_close'] * 100
    prices_df_clean = prices_df[prices_df['date_str'].isin(trading_dates)].dropna(subset=['return_pct'])
    
    # Prefetch index data
    index_df = con.execute("""
        SELECT date, close AS Close
        FROM stock_prices
        WHERE symbol = 'NSEI'
        ORDER BY date
    """).fetchdf()
    if not index_df.empty:
        index_df['Date'] = pd.to_datetime(index_df['date']).dt.tz_localize(None)
        index_df = index_df.set_index('Date')
        
    # Get Nifty 100 symbols list (which was the default scanned universe for BTST)
    nifty_100_symbols = set(get_stock_universe(max_stocks=100))
    
    # We will track misses
    missed_btst_signals = []
    rule_failures = {
        "uptrend": 0,
        "near_high": 0,
        "vol_spike": 0,
        "strong_candle": 0,
        "rsi_momentum": 0,
        "index_filter": 0,
        "universe_limit": 0
    }
    
    evaluated_gainers_count = 0
    
    for d_str in trading_dates:
        day_prices = prices_df_clean[prices_df_clean['date_str'] == d_str]
        if day_prices.empty:
            continue
            
        # Top 5 daily gainers
        top_5 = day_prices.sort_values(by='return_pct', ascending=False).head(5)
        
        for idx, row in top_5.iterrows():
            symbol = row['symbol']
            ret = row['return_pct']
            evaluated_gainers_count += 1
            
            # Slice historical data up to this day for evaluation
            # Query full history of this symbol to ensure sufficient rows for indicators (like SMA50)
            symbol_history = con.execute("""
                SELECT date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
                FROM stock_prices
                WHERE symbol = ? AND date <= ?
                ORDER BY date
            """, [symbol, d_str + " 23:59:59"]).fetchdf()
            
            if len(symbol_history) < 60:
                continue
                
            symbol_history['date_str'] = pd.to_datetime(symbol_history['date']).dt.strftime('%Y-%m-%d')
            symbol_history['Date'] = pd.to_datetime(symbol_history['date']).dt.tz_localize(None)
            symbol_history = symbol_history.set_index('Date').sort_index()
            
            # Evaluate using BTST logic
            try:
                # We replicate BTST evaluation to see which rules failed
                df_ind = symbol_history.copy()
                import ta
                df_ind["sma20"] = ta.trend.sma_indicator(df_ind["Close"], window=20)
                df_ind["sma50"] = ta.trend.sma_indicator(df_ind["Close"], window=50)
                df_ind["vol_avg20"] = df_ind["Volume"].shift(1).rolling(20).mean()
                df_ind["rsi14"] = ta.momentum.rsi(df_ind["Close"], window=14)
                df_ind = df_ind.dropna()
                
                if len(df_ind) < 5:
                    continue
                    
                last = df_ind.iloc[-1]
                
                # Check individual rules
                uptrend = last["Close"] > last["sma20"] and last["Close"] > last["sma50"]
                near_high = last["Close"] >= 0.993 * last["High"]
                
                vol_ratio = last["Volume"] / last["vol_avg20"]
                vol_spike = vol_ratio >= 2.0
                
                today_ret = (last["Close"] - last["Open"]) / last["Open"] * 100
                strong_candle = last["Close"] > last["Open"] and today_ret >= 1.5
                
                rsi_momentum = 55 <= last["rsi14"] <= 78
                
                index_ok = True
                if index_df is not None and not index_df.empty:
                    idx_slice = index_df.loc[index_df.index <= df_ind.index[-1]]
                    if not idx_slice.empty and len(idx_slice) >= 50:
                        idx_close = idx_slice["Close"]
                        idx_sma50 = ta.trend.sma_indicator(idx_close, window=50)
                        if not idx_sma50.empty and df_ind.index[-1] in idx_sma50.index:
                            index_ok = idx_close.loc[df_ind.index[-1]] > idx_sma50.loc[df_ind.index[-1]]
                            
                # Check if it passes all BTST criteria
                if uptrend and near_high and vol_spike and strong_candle and rsi_momentum and index_ok:
                    # Yes! It was a valid BTST setup candidate on the day before its big move.
                    # Was it in Nifty 100?
                    is_in_nifty100 = symbol in nifty_100_symbols
                    
                    missed_btst_signals.append({
                        "date": d_str,
                        "symbol": symbol,
                        "return_pct": ret,
                        "close": last["Close"],
                        "rsi": last["rsi14"],
                        "vol_ratio": vol_ratio,
                        "is_in_nifty100": is_in_nifty100,
                        "miss_reason": "Universe Limit (Nifty 200 stock)" if not is_in_nifty100 else "Alert Capping / Capped"
                    })
                    if not is_in_nifty100:
                        rule_failures["universe_limit"] += 1
                else:
                    # Count failed rules
                    if not uptrend: rule_failures["uptrend"] += 1
                    if not near_high: rule_failures["near_high"] += 1
                    if not vol_spike: rule_failures["vol_spike"] += 1
                    if not strong_candle: rule_failures["strong_candle"] += 1
                    if not rsi_momentum: rule_failures["rsi_momentum"] += 1
                    if not index_ok: rule_failures["index_filter"] += 1
            except Exception as e:
                print(f"Error evaluating {symbol} on {d_str}: {e}")
                
    con.close()
    
    print("\n" + "=" * 80)
    print("                     BTST MISS ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Daily Top Gainers Analyzed (top 5 per day): {evaluated_gainers_count}")
    print(f"Daily Top Gainers that MET BTST Criteria but were Missed: {len(missed_btst_signals)}")
    print("-" * 80)
    
    # Print rule failure statistics
    print("Reasons why other top gainers did NOT meet BTST criteria (multiple can fail):")
    print(f" - Price not in SMA20/50 Uptrend           : {rule_failures['uptrend']} times")
    print(f" - Did not close near high (within 0.7%)  : {rule_failures['near_high']} times")
    print(f" - Volume spike was too low (< 2x average) : {rule_failures['vol_spike']} times")
    print(f" - Weak daily candle (return < 1.5%)       : {rule_failures['strong_candle']} times")
    print(f" - RSI outside momentum zone (55 - 78)     : {rule_failures['rsi_momentum']} times")
    print(f" - Nifty 50 was in downtrend (< SMA50)     : {rule_failures['index_filter']} times")
    print(f" - Excluded by Universe (Nifty 200 stock)  : {rule_failures['universe_limit']} times")
    print("-" * 80)
    
    if not missed_btst_signals:
        print("No valid BTST setups were missed among the daily top gainers in this period.")
    else:
        print("Missed BTST Candidates details (top gainers that met all criteria):")
        df_misses = pd.DataFrame(missed_btst_signals)
        print(df_misses.to_string(index=False, columns=['date', 'symbol', 'return_pct', 'close', 'rsi', 'vol_ratio', 'miss_reason']))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
