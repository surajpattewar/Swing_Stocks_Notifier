#!/usr/bin/env python3

# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
"""
Script to clear old price data and download 5 years of daily stock prices
for 205 symbols and the NSEI benchmark.
"""
import os
import sys
import logging
import shutil

# Add the workspace root to sys.path


from config import config
from stock_universe import get_stock_universe
from data_ingestion import ingest_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_5y_data")

def main():
    db_path = config.DUCKDB_PATH
    
    # 1. Remove old DuckDB file
    if os.path.exists(db_path):
        logger.info(f"Removing existing database file at {db_path} to clear all old data...")
        try:
            os.remove(db_path)
            logger.info("Successfully removed old database.")
        except Exception as e:
            logger.error(f"Failed to remove database file: {e}")
            return 1
            
    # 2. Get the stock universe
    # We want 205 symbols. We ask get_stock_universe for 205 stocks.
    symbols = get_stock_universe(max_stocks=205)
    
    # Ensure ^NSEI or NSEI is present
    if "^NSEI" not in symbols and "NSEI" not in symbols:
        symbols.append("^NSEI")
        
    logger.info(f"Retrieved stock universe with {len(symbols)} symbols (including index).")
    
    # 3. Download 5 years of daily prices
    logger.info("Starting historical download for the last 5 years from Yahoo Finance...")
    ingest_history(symbols, db_path=db_path, period="5y")
    logger.info("Finished ingesting 5 years of historical data!")
    return 0

if __name__ == "__main__":
    main()