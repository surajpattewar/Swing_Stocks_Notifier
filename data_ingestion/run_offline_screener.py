
# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
import os
import duckdb
import pandas as pd
from datetime import datetime
from config import config
from screener import evaluate, Candidate

def main():
    db_path = config.DUCKDB_PATH
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
    print(f"Latest database date: {latest_date}")
    
    # Load all symbols
    symbols_rows = con.execute("SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol").fetchall()
    symbols = [r[0] for r in symbols_rows if r[0] != 'NSEI']
    
    # Load index data
    index_raw = con.execute(
        """
        SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
               close AS Close
        FROM stock_prices
        WHERE symbol = 'NSEI'
        ORDER BY date
        """
    ).fetchdf()
    index_df = pd.DataFrame()
    if not index_raw.empty:
        index_raw["Date"] = pd.to_datetime(index_raw["Date"])
        index_df = index_raw.set_index("Date")
        
    candidates = []
    
    print(f"Screening {len(symbols)} symbols as of {latest_date}...")
    for symbol in symbols:
        try:
            # Load price history up to latest_date
            df = con.execute(
                """
                SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                       open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
                FROM stock_prices
                WHERE symbol = ? AND CAST(timezone('Asia/Kolkata', date) AS DATE) <= ?
                ORDER BY date
                """,
                [symbol, latest_date]
            ).fetchdf()
            
            if df.empty or len(df) < 110:
                continue
                
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            
            # Evaluate using our optimized evaluate function
            cand = evaluate(symbol, df, skip_fundamental=True, index_df=index_df)
            if cand.score >= config.MIN_SCORE:
                candidates.append(cand)
        except Exception as e:
            continue
            
    con.close()
    
    candidates.sort(key=lambda c: (c.score, c.adx), reverse=True)
    
    print("\n" + "=" * 80)
    print(f"                      OFFLINE SCREENER RESULTS (Score >= {config.MIN_SCORE})")
    print("=" * 80)
    print(f"As of: {latest_date}")
    print(f"Found: {len(candidates)} stocks")
    print("-" * 80)
    
    if not candidates:
        print("No qualifying stocks found.")
    else:
        for cand in candidates[:config.TOP_N_ALERTS]:
            print(cand.to_line())
            print()
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
