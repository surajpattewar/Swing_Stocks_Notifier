import os
import duckdb
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

# Import local db tracker modules
from db_tracker import init_db, save_daily_results, update_open_positions, track_daily_progress
from config import config
from screener import evaluate, Candidate

def main():
    # Initialize the target database (Google Sheets)
    init_db()
    
    # Clear previous sheets before starting a clean run
    from db_tracker import clear_sheets
    print("Clearing all previous data in Google Sheets...")
    clear_sheets()
    
    # Ingest the latest price updates from yfinance before backfilling
    from data_ingestion import ingest_deltas
    from stock_universe import get_stock_universe
    print("Ingesting latest stock price data from yfinance...")
    scan_symbols = get_stock_universe(max_stocks=config.BACKTEST_MAX_STOCKS)
    ingest_symbols = scan_symbols + ["^NSEI"]
    ingest_deltas(ingest_symbols)
    
    # Connect to the local DuckDB price index
    duck_db_path = "data/duckdb/screener_data.duckdb"
    if not os.path.exists(duck_db_path):
        print(f"Error: Local DuckDB cache {duck_db_path} not found.")
        return
        
    con_duck = duckdb.connect(duck_db_path, read_only=True)
    
    # Get the latest 44 available days (approx 2 months) in Nifty index
    dates_raw = con_duck.execute(
        """
        SELECT DISTINCT CAST(timezone('Asia/Kolkata', date) AS DATE) AS d 
        FROM stock_prices 
        WHERE symbol = 'NSEI' 
        ORDER BY d DESC 
        LIMIT 44
        """
    ).fetchall()
    
    if not dates_raw:
        print("Error: No dates found in DuckDB.")
        con_duck.close()
        return
        
    # Sort dates chronologically
    dates = sorted([r[0] for r in dates_raw])
    print(f"Chronological dates to backfill: {[str(d) for d in dates]}")
    
    # Load all symbols
    symbols_rows = con_duck.execute("SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol").fetchall()
    symbols = [r[0] for r in symbols_rows if r[0] != 'NSEI']
    print(f"Total symbols found in DuckDB: {len(symbols)}")
    
    # Load Nifty index data
    index_raw = con_duck.execute(
        """
        SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
               close AS Close
        FROM stock_prices
        WHERE symbol = 'NSEI'
        ORDER BY date
        """
    ).fetchdf()
    index_raw["Date"] = pd.to_datetime(index_raw["Date"])
    index_df = index_raw.set_index("Date")
    if index_df.index.tz is not None:
        index_df.index = index_df.index.tz_localize(None)
        
    # Google Sheets initialization is handled by init_db()
    
    for D in dates:
        D_str = str(D)
        print(f"\n" + "="*80)
        print(f" Simulating daily run for: {D_str}")
        print("="*80)
        
        # 1. Run the screener on Nifty 100 stocks as of date D
        candidates = []
        index_slice = index_df.loc[index_df.index <= pd.Timestamp(D)]
        
        for symbol in symbols:
            try:
                # Load price history up to date D
                df = con_duck.execute(
                    """
                    SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                           open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
                    FROM stock_prices
                    WHERE symbol = ? AND CAST(timezone('Asia/Kolkata', date) AS DATE) <= ?
                    ORDER BY date
                    """,
                    [symbol, D]
                ).fetchdf()
                
                if df.empty or len(df) < 110:
                    continue
                    
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                    
                cand = evaluate(symbol, df, skip_fundamental=True, index_df=index_slice)
                if cand.score >= config.MIN_SCORE:
                    candidates.append(cand)
            except Exception as e:
                continue
                
        candidates.sort(key=lambda c: (c.score, c.adx), reverse=True)
        candidates = candidates[:config.TOP_N_ALERTS]
        print(f"Found {len(candidates)} candidates (capped to top {config.TOP_N_ALERTS}) with score >= {config.MIN_SCORE} as of {D_str}.")
        
        # 2. Persist daily results for date D
        try:
            save_daily_results(D_str, candidates)
            update_open_positions(D_str, candidates)
            track_daily_progress(D_str)
            print(f"  - Successfully saved daily candidates, updated positions, and tracked progress for {D_str} in Google Sheets.")
        except Exception as e:
            print(f"Error on date {D_str}: {e}")
            
    con_duck.close()
    print("\nBackfill complete! Check daily_results, open_positions, and position_progress tables.")

if __name__ == "__main__":
    main()
