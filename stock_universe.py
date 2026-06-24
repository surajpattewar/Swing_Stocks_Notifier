"""
Builds the list of NSE stock symbols to scan (in yfinance format, e.g. RELIANCE.NS).
Tries to pull the live NSE equity list; falls back to a static, liquid-stock list
if NSE blocks the request (common from cloud IPs) so the bot never just dies.
"""
import io
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NIFTY_500_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
NIFTY_200_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"
NIFTY_100_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"

# Fallback: ~100 liquid, large/mid-cap NSE stocks. Edit this list freely to
# match your own watchlist/universe.
FALLBACK_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC",
    "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TMPV", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"
]


def _format(symbol: str) -> str:
    return f"{symbol.strip().upper()}.NS"


def get_stock_universe(max_stocks: int = 200, url: str=None, no_of_stocks:int=100) -> list:
    """
    Returns a list of yfinance-formatted NSE symbols, e.g. ['RELIANCE.NS', ...].
    """
    try:
        if not url:
            url = f"https://nsearchives.nseindia.com/content/indices/ind_nifty{no_of_stocks}list.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.upper().str.strip()
        symbols = df["SYMBOL"].dropna().astype(str).tolist()
        symbols.append("^NSEI")
        if not symbols:
            raise ValueError("Empty symbol list from NSE")
        logger.info("Fetched %d symbols from live NSE list", len(symbols))
    except Exception as e:
        logger.warning("Live NSE list fetch failed (%s); using fallback list", e)
        symbols = FALLBACK_SYMBOLS

    symbols = symbols[:max_stocks]
    return [_format(s) for s in symbols]
