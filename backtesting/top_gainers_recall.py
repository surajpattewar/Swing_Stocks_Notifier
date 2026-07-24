
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
from btst_screener import evaluate_btst
from backtest import get_local_symbols

def main():
    db_path = config.DUCKDB_PATH
    con = duckdb.connect(db_path, read_only=True)
    symbols = get_local_symbols(db_path, max_stocks=150)
    symbols = [s for s in symbols if s != 'NSEI']
    
    # Load Nifty index
    index_raw = con.execute("SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date, close AS Close FROM stock_prices WHERE symbol = 'NSEI' ORDER BY date").fetchdf()
    index_df = pd.DataFrame()
    if not index_raw.empty:
        index_raw["Date"] = pd.to_datetime(index_raw["Date"])
        index_df = index_raw.set_index("Date")
        
    end_date = pd.Timestamp("2026-07-22").date()
    start_date = end_date - timedelta(days=30)
    
    print(f"============================================================")
    print(f"   BTST MODEL RECALL ANALYSIS: LAST 30 DAYS TOP GAINERS")
    print(f"   Period: {start_date} to {end_date}")
    print(f"============================================================\n")
    
    # Pre-load all stock prices to memory for fast evaluation
    print("Loading price history for all symbols...")
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
    
    # Find all trading dates in the last 30 days
    if not stock_data:
        print("No stock data loaded.")
        return
        
    sample_df = list(stock_data.values())[0]
    trading_dates = sample_df.index[(sample_df.index.date >= start_date) & (sample_df.index.date <= end_date)].tolist()
    trading_dates = sorted(trading_dates)
    
    total_top_gainers = 0
    correctly_predicted = 0
    predictions_log = []
    
    # We define a "Top Gainer" as a stock that gained >= 2.0% on day X
    MIN_GAINER_PCT = 2.0
    
    # Loop from the second date to the last
    for idx in range(1, len(trading_dates)):
        prev_date = trading_dates[idx - 1]
        curr_date = trading_dates[idx]
        
        # 1. Identify the top gainers on curr_date
        curr_gainers = []
        for symbol, df in stock_data.items():
            if prev_date in df.index and curr_date in df.index:
                prev_row = df.loc[prev_date]
                curr_row = df.loc[curr_date]
                
                # Check return on current date (Open to Close return)
                ret_pct = (curr_row["Close"] - curr_row["Open"]) / curr_row["Open"] * 100
                if ret_pct >= MIN_GAINER_PCT:
                    curr_gainers.append((symbol, ret_pct, curr_row["Close"]))
                    
        if not curr_gainers:
            continue
            
        # 2. Check if our BTST screener triggered on prev_date for these gainers
        for symbol, ret_pct, close in curr_gainers:
            total_top_gainers += 1
            
            df = stock_data[symbol]
            df_slice = df.loc[df.index <= prev_date]
            
            if len(df_slice) < 50:
                continue
                
            try:
                # Run the BTST screener
                cand = evaluate_btst(symbol, df_slice, index_df=index_df)
                if cand is not None:
                    correctly_predicted += 1
                    predictions_log.append({
                        "symbol": symbol,
                        "trigger_date": prev_date.date(),
                        "gain_date": curr_date.date(),
                        "gain_pct": ret_pct,
                        "close": close
                    })
            except Exception:
                continue
                
    # Display results
    recall_rate = (correctly_predicted / total_top_gainers * 100) if total_top_gainers > 0 else 0.0
    
    print("\n==============================================")
    print("                RECALL SUMMARY")
    print("==============================================")
    print(f"Total Top Gainers (gaining >= {MIN_GAINER_PCT}%): {total_top_gainers}")
    print(f"Correctly Identified on Previous Day: {correctly_predicted}")
    print(f"Recall Accuracy Rate: {recall_rate:.2f}%")
    print("==============================================")
    
    if predictions_log:
        print("\nExamples of Successfully Predicted Top Gainers:")
        log_df = pd.DataFrame(predictions_log)
        # Sort by gain percentage descending
        log_df = log_df.sort_values(by="gain_pct", ascending=False).head(15)
        for _, row in log_df.iterrows():
            print(f"• {row['symbol'].replace('.NS', '')}: Triggered on {row['trigger_date']} -> Gained {row['gain_pct']:+.2f}% on {row['gain_date']} (CMP: ₹{row['close']:.2f})")
    else:
        print("\nNo successful predictions found in log.")

if __name__ == "__main__":
    main()
