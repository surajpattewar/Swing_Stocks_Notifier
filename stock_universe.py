"""
Builds the list of NSE stock symbols to scan (in yfinance format, e.g. RELIANCE.NS).
Tries to pull the live NSE equity list; falls back to a static, liquid-stock list
if NSE blocks the request (common from cloud IPs) so the bot never just dies.
"""
import io
import logging

import pandas as pd
import requests
from config import config

logger = logging.getLogger(__name__)

NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NIFTY_500_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
NIFTY_200_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"
NIFTY_100_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"

# Fallback: full Nifty 200 symbols list to ensure coverage when live NSE fetch fails.
FALLBACK_SYMBOLS = [
    '360ONE', 'ABB', 'ABCAPITAL', 'ADANIENSOL', 'ADANIENT', 'ADANIGREEN', 'ADANIPORTS', 'ADANIPOWER', 'ALKEM', 'AMBUJACEM',
    'APLAPOLLO', 'APOLLOHOSP', 'ASHOKLEY', 'ASIANPAINT', 'ASTRAL', 'ATGL', 'AUBANK', 'AUROPHARMA', 'AXISBANK', 'BAJAJ-AUTO',
    'BAJAJFINSV', 'BAJAJHLDNG', 'BAJFINANCE', 'BANKBARODA', 'BANKINDIA', 'BDL', 'BEL', 'BHARATFORG', 'BHARTIARTL', 'BHEL',
    'BIOCON', 'BLUESTARCO', 'BOSCHLTD', 'BPCL', 'BRITANNIA', 'BSE', 'CANBK', 'CGPOWER', 'CHOLAFIN', 'CIPLA',
    'COALINDIA', 'COCHINSHIP', 'COFORGE', 'COLPAL', 'CONCOR', 'COROMANDEL', 'CUMMINSIND', 'DABUR', 'DIVISLAB', 'DIXON',
    'DLF', 'DMART', 'DREDGECORP', 'DRREDDY', 'EICHERMOT', 'ENRIN', 'ETERNAL', 'EXIDEIND', 'FEDERALBNK', 'FORTIS',
    'GAIL', 'GLENMARK', 'GMRAIRPORT', 'GODFRYPHLP', 'GODREJCP', 'GODREJPROP', 'GRASIM', 'GROWW', 'GVT&D', 'HAL',
    'HAVELLS', 'HCLTECH', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDPETRO', 'HINDUNILVR', 'HINDZINC',
    'HSCL', 'HUDCO', 'HYUNDAI', 'ICICIAMC', 'ICICIBANK', 'ICICIGI', 'IDEA', 'IDFCFIRSTB', 'INDHOTEL', 'INDIANB',
    'INDIGO', 'INDUSINDBK', 'INDUSTOWER', 'INFY', 'IOC', 'IRCTC', 'IREDA', 'IRFC', 'ITC', 'JINDALSTEL',
    'JIOFIN', 'JSWENERGY', 'JSWSTEEL', 'JUBLFOOD', 'KALYANKJIL', 'KEI', 'KOTAKBANK', 'KPITTECH', 'LAURUSLABS', 'LENSKART',
    'LGEINDIA', 'LICHSGFIN', 'LODHA', 'LT', 'LTF', 'LTM', 'LUPIN', 'M&M', 'M&MFIN', 'MANKIND',
    'MARICO', 'MARUTI', 'MAXHEALTH', 'MAZDOCK', 'MCX', 'MFSL', 'MOTHERSON', 'MOTILALOFS', 'MPHASIS', 'MRF',
    'MUTHOOTFIN', 'NATIONALUM', 'NAUKRI', 'NESTLEIND', 'NHPC', 'NMDC', 'NTPC', 'NYKAA', 'OBEROIRLTY', 'OFSS',
    'OIL', 'ONGC', 'PAGEIND', 'PATANJALI', 'PAYTM', 'PERSISTENT', 'PFC', 'PHOENIXLTD', 'PIDILITIND', 'PIIND',
    'PNB', 'POLICYBZR', 'POLYCAB', 'POWERGRID', 'POWERINDIA', 'PREMIERENE', 'PRESTIGE', 'RADICO', 'RECLTD', 'RELIANCE',
    'RVNL', 'SAIL', 'SBICARD', 'SBILIFE', 'SBIN', 'SHREECEM', 'SHRIRAMFIN', 'SIEMENS', 'SOLARINDS', 'SRF',
    'SUNPHARMA', 'SUPREMEIND', 'SUZLON', 'SWIGGY', 'TATACAP', 'TATACHEM', 'TATACOMM', 'TATACONSUM', 'TATAELXSI', 'TATAINVEST',
    'TATAPOWER', 'TATASTEEL', 'TCS', 'TECHM', 'TIINDIA', 'TITAN', 'TMCV', 'TMPV', 'TORNTPHARM', 'TRENT',
    'TVSMOTOR', 'ULTRACEMCO', 'UNIONBANK', 'UNITDSPR', 'UPL', 'VAML', 'VBL', 'VEDL', 'VMM', 'VOGL',
    'VOLTAS', 'WAAREEENER', 'WIPRO', 'YESBANK', 'ZYDUSLIFE'
]


def _format(symbol: str) -> str:
    return f"{symbol.strip().upper()}.NS"


def get_stock_universe(max_stocks: int = 250, url: str=None, no_of_stocks:int=200) -> list:
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
        if not symbols:
            raise ValueError("Empty symbol list from NSE")
        logger.info("Fetched %d symbols from live NSE list", len(symbols))
    except Exception as e:
        logger.warning("Live NSE list fetch failed (%s); using fallback list", e)
        symbols = FALLBACK_SYMBOLS

    symbols = [_format(s) for s in symbols[:max_stocks]]
    symbols.extend(config.ADDITIONAL_SYMBOLS)
    symbols = list(dict.fromkeys(symbols))

    # symbols.append("^NSEI")
    return symbols
