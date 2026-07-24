import os
import sys
import duckdb
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Ensure the root of the project is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environmental variables
load_dotenv()

# Import local db tracker modules
from tuning_scripts.tuning_db_tracker import init_db_tuning, save_tuning_signals, clear_tuning_sheet
from config import config
from screener import evaluate, _add_indicators
from stock_universe import get_stock_universe
from data_ingestion import ingest_deltas

def main():
    print("Initializing Google Sheets database for tuning...")
    init_db_tuning()
    
    print("Clearing all previous data in tuning_sheet...")
    clear_tuning_sheet()
    
    print("Fetching Nifty 200 stock universe...")
    scan_symbols = get_stock_universe(max_stocks=500, no_of_stocks=200)
    print(f"Total Nifty 200 symbols to scan: {len(scan_symbols)}")
    
    # Ingest the latest price updates from yfinance before backfilling
    print("Ingesting latest stock price data from yfinance for Nifty 200 stocks...")
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
        
    for D in dates:
        D_str = str(D)
        print(f"\n" + "="*80)
        print(f" Simulating daily run for: {D_str}")
        print("="*80)
        
        candidates_data = []
        index_slice = index_df.loc[index_df.index <= pd.Timestamp(D)]
        
        for symbol in scan_symbols:
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
                    
                # Calculate indicators to extract ATR
                df_inds = _add_indicators(df.copy())
                atr_val = float(df_inds.iloc[-1]["atr"]) if not df_inds.empty and "atr" in df_inds.columns else 0.0
                
                cand = evaluate(symbol, df, skip_fundamental=True, index_df=index_slice)
                
                # We save all signals with score >= 3 for tuning purposes
                if cand.score >= 3:
                    candidates_data.append({
                        "symbol": symbol,
                        "setup_type": cand.setup_type,
                        "score": cand.score,
                        "close": cand.close,
                        "stop_loss": cand.stop_loss,
                        "target": cand.target,
                        "reasons": "; ".join(cand.reasons) if isinstance(cand.reasons, list) else str(cand.reasons),
                        "beta": cand.beta,
                        "adx": cand.adx,
                        "atr": atr_val
                    })
            except Exception as e:
                continue
                
        if candidates_data:
            print(f"Saving {len(candidates_data)} signals for {D_str} to Google Sheets...")
            save_tuning_signals(D_str, candidates_data)
            
    con_duck.close()
    print("\nBackfill complete! Check tuning_sheet table.")

if __name__ == "__main__":
    main()
