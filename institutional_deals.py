"""
institutional_deals.py
======================
Downloads and parses daily Bulk and Block Deals CSV files from the NSE archives.
Aggregates buys/sells per stock to identify institutional volume catalysts.
"""

import os
import logging
import subprocess
import shutil
import pandas as pd

logger = logging.getLogger(__name__)

BULK_URL = "https://nsearchives.nseindia.com/content/equities/bulk.csv"
BLOCK_URL = "https://nsearchives.nseindia.com/content/equities/block.csv"

_DEALS_CACHE = None

def download_file(url: str, output_path: str) -> bool:
    """Download a file using curl (via subprocess) with a urllib fallback."""
    try:
        # 1. Try curl
        if shutil.which("curl"):
            cmd = [
                "curl", "-L", "-s", "-o", output_path, url,
                "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully downloaded {url} via curl.")
                return True
                
        # 2. Try urllib fallback
        import urllib.request
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(output_path, "wb") as f:
                f.write(response.read())
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully downloaded {url} via urllib.")
            return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
    return False

def parse_deals_csv(filepath: str, deal_type: str) -> pd.DataFrame:
    """Reads and parses an NSE bulk or block deals CSV, cleaning up column headers."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return pd.DataFrame()
            
        # Clean and normalize columns to uppercase
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Standard columns we need
        col_map = {}
        for col in df.columns:
            if "SYMBOL" in col:
                col_map["SYMBOL"] = col
            elif "BUY" in col or "SELL" in col:
                col_map["BUY_SELL"] = col
            elif "QUANTITY" in col or "QTY" in col:
                col_map["QUANTITY"] = col
            elif "PRICE" in col:
                col_map["PRICE"] = col
            elif "CLIENT" in col:
                col_map["CLIENT"] = col
            elif "DATE" in col:
                col_map["DATE"] = col
                
        # Rename identified columns to standardized names
        reverse_map = {v: k for k, v in col_map.items()}
        df = df.rename(columns=reverse_map)
        
        # Ensure standard columns exist
        required = ["SYMBOL", "BUY_SELL", "QUANTITY", "PRICE"]
        for r in required:
            if r not in df.columns:
                logger.warning(f"Required column {r} missing in {deal_type} deal CSV. Columns found: {df.columns.tolist()}")
                return pd.DataFrame()
                
        # Clean symbol names
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
        df["BUY_SELL"] = df["BUY_SELL"].astype(str).str.strip().str.upper()
        
        # Clean quantity and price values (handling string formatting/commas)
        df["QUANTITY"] = pd.to_numeric(df["QUANTITY"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
        df["PRICE"] = pd.to_numeric(df["PRICE"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
        
        df["DEAL_TYPE"] = deal_type
        return df
        
    except Exception as e:
        logger.warning(f"Error parsing {deal_type} deals CSV {filepath}: {e}")
        return pd.DataFrame()

def load_institutional_deals() -> pd.DataFrame:
    """
    Downloads and compiles today's Bulk and Block deals.
    If downloads fail, it falls back to local bulk.csv / block.csv if they exist.
    """
    global _DEALS_CACHE
    if _DEALS_CACHE is not None:
        return _DEALS_CACHE
        
    bulk_path = "bulk.csv"
    block_path = "block.csv"
    
    # 1. Download files
    download_file(BULK_URL, bulk_path)
    download_file(BLOCK_URL, block_path)
    
    # 2. Parse datasets
    bulk_df = parse_deals_csv(bulk_path, "BULK")
    block_df = parse_deals_csv(block_path, "BLOCK")
    
    # Merge datasets
    merged = pd.DataFrame()
    if not bulk_df.empty and not block_df.empty:
        merged = pd.concat([bulk_df, block_df], ignore_index=True)
    elif not bulk_df.empty:
        merged = bulk_df
    elif not block_df.empty:
        merged = block_df
        
    _DEALS_CACHE = merged
    if not merged.empty:
        logger.info(f"Loaded {len(merged)} total institutional deals.")
    else:
        logger.info("No institutional deals loaded (either CSVs empty or download failed).")
    return merged

def get_deal_catalyst(symbol: str) -> dict:
    """
    Analyzes the bulk and block deals for a given symbol.
    Returns:
      - has_buy_deal: True if there was a net positive institutional purchase
      - has_sell_deal: True if there was a net positive institutional sale
      - net_shares: Net quantity bought (negative if sold)
      - details: String describing the main players and quantities
    """
    result = {
        "has_buy_deal": False,
        "has_sell_deal": False,
        "net_shares": 0,
        "details": ""
    }
    
    df = load_institutional_deals()
    if df.empty:
        return result
        
    clean_sym = symbol.replace(".NS", "").strip()
    stock_deals = df[df["SYMBOL"] == clean_sym]
    if stock_deals.empty:
        return result
        
    net_shares = 0
    buys = []
    sells = []
    
    for _, row in stock_deals.iterrows():
        action = row["BUY_SELL"]
        qty = row["QUANTITY"]
        price = row["PRICE"]
        client = row.get("CLIENT", "Unknown Client")
        deal_type = row["DEAL_TYPE"]
        
        if action == "BUY":
            net_shares += qty
            buys.append(f"{client} ({deal_type} Buy {int(qty):,} shares @ ₹{price:.1f})")
        elif action == "SELL":
            net_shares -= qty
            sells.append(f"{client} ({deal_type} Sell {int(qty):,} shares @ ₹{price:.1f})")
            
    result["net_shares"] = int(net_shares)
    
    if net_shares > 0:
        result["has_buy_deal"] = True
        result["details"] = "; ".join(buys)
    elif net_shares < 0:
        result["has_sell_deal"] = True
        result["details"] = "; ".join(sells)
        
    return result
