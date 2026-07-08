"""
news_analyzer.py
================
Fetches and analyzes recent news headlines and calendar events for a stock using yfinance and NSE API.
Acts as a secondary filter to avoid event risk (upcoming earnings) and score news sentiment.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests

logger = logging.getLogger(__name__)

# Financial sentiment keywords (calibrated for headlines)
BULLISH_KEYWORDS = {
    "buy", "outperform", "growth", "rise", "profit", "raises", "expansion",
    "partnership", "deal", "win", "bullish", "target raised", "upgrade",
    "highest", "record", "gains", "recovery", "surge", "beats", "dividend"
}

BEARISH_KEYWORDS = {
    "sell", "underperform", "slump", "drop", "loss", "probe", "fraud",
    "investigation", "fine", "default", "pledge", "downgrade", "deficit",
    "slashed", "falls", "penalty", "scam", "cuts", "debt", "crisis", "warns"
}

_NSE_CALENDAR_CACHE = None

def clean_event_calendar_csv(csv_text: str) -> str:
    if not csv_text:
        return csv_text

    # 1. Remove BOM if present
    if csv_text.startswith('\ufeff'):
        csv_text = csv_text[1:]

    # 2. Normalize line endings
    csv_text = csv_text.replace('\r\n', '\n')

    # 3. Clean the headers
    # Corrupt headers look like: "SYMBOL \n","COMPANY \n","PURPOSE \n","DETAILS \n","DATE \n"\n
    import re
    pattern = r'^"SYMBOL\s*\n?","COMPANY\s*\n?","PURPOSE\s*\n?","DETAILS\s*\n?","DATE\s*\n?"\n?'
    clean_headers = '"SYMBOL","COMPANY","PURPOSE","DETAILS","DATE"\n'
    
    cleaned_text = re.sub(pattern, clean_headers, csv_text, flags=re.IGNORECASE)
    
    # Fallback if regex did not match (e.g. slight formatting changes)
    if cleaned_text == csv_text:
        first_newline_idx = csv_text.find('\n', 200)
        if first_newline_idx != -1:
            header_part = csv_text[:first_newline_idx]
            rest_part = csv_text[first_newline_idx:]
            
            header_part = header_part.replace('\n', '').replace('\r', '')
            if "SYMBOL" in header_part.upper() and "DATE" in header_part.upper():
                cleaned_text = '"SYMBOL","COMPANY","PURPOSE","DETAILS","DATE"' + rest_part

    return cleaned_text

def download_event_calendar_online() -> str:
    import os
    url = "https://www.nseindia.com/api/event-calendar?index=equities&csv=true"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar",
        "Connection": "keep-alive"
    }

    # Method 1: Try curl_cffi (impersonate browser to bypass WAF/Cloudflare)
    try:
        from curl_cffi import requests as requests_cffi
        logger.info("Attempting online event calendar fetch using curl_cffi...")
        session = requests_cffi.Session()
        # Visit homepage to establish session cookies
        session.get("https://www.nseindia.com", headers=headers, impersonate="chrome120", timeout=15)
        resp = session.get(url, headers=headers, impersonate="chrome120", timeout=15)
        if resp.status_code == 200 and resp.text:
            if "SYMBOL" in resp.text.upper():
                logger.info("Successfully fetched calendar via curl_cffi.")
                return resp.text
    except Exception as e:
        logger.warning(f"curl_cffi fetch failed: {e}")

    # Method 2: Try standard requests session with homepage cookie initialization
    try:
        import requests
        logger.info("Attempting online event calendar fetch using standard requests...")
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200 and resp.text:
            if "SYMBOL" in resp.text.upper():
                logger.info("Successfully fetched calendar via standard requests.")
                return resp.text
    except Exception as e:
        logger.warning(f"Standard requests fetch failed: {e}")

    # Method 3: Try subprocess curl mimicking a real browser
    try:
        import subprocess
        import shutil
        if shutil.which("curl"):
            logger.info("Attempting online event calendar fetch using subprocess curl...")
            cookie_jar = "nse_cookies.txt"
            # 1. Fetch homepage to get cookies
            subprocess.run([
                "curl", "-s", "-L", "-o", "/dev/null",
                "-c", cookie_jar,
                "-H", f"User-Agent: {headers['User-Agent']}",
                "-H", f"Referer: {headers['Referer']}",
                "https://www.nseindia.com"
            ], timeout=10)
            # 2. Fetch the event calendar API using those cookies
            cmd = [
                "curl", "-s", "-L",
                "-b", cookie_jar,
                "-H", f"User-Agent: {headers['User-Agent']}",
                "-H", f"Referer: {headers['Referer']}",
                url
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            # Clean up cookie jar
            if os.path.exists(cookie_jar):
                try:
                    os.remove(cookie_jar)
                except Exception:
                    pass
            if res.returncode == 0 and res.stdout:
                if "SYMBOL" in res.stdout.upper():
                    logger.info("Successfully fetched calendar via subprocess curl.")
                    return res.stdout
    except Exception as e:
        logger.warning(f"Subprocess curl fetch failed: {e}")

    raise ValueError("All online download methods failed.")

def get_nse_corporate_events() -> pd.DataFrame:
    """
    Fetches the full corporate filings event calendar from NSE India.
    Attempts to download the latest calendar online using multiple fallback methods
    (curl_cffi, standard requests, subprocess curl). Saves the downloaded file locally
    to event-calendar.csv as a cache, and falls back to loading this local CSV file
    if online fetching fails.
    """
    global _NSE_CALENDAR_CACHE
    if _NSE_CALENDAR_CACHE is not None:
        return _NSE_CALENDAR_CACHE

    import os
    import glob
    from io import StringIO

    local_path = "event-calendar.csv"

    # 1. Attempt online fetch first
    csv_text = None
    try:
        csv_text = download_event_calendar_online()
    except Exception as ex:
        logger.warning(f"Online event calendar download failed: {ex}. Using local fallbacks...")

    if csv_text:
        try:
            # Clean headers before parsing or saving
            csv_text = clean_event_calendar_csv(csv_text)
            df = pd.read_csv(StringIO(csv_text))
            df.columns = [c.strip().upper() for c in df.columns]
            if "SYMBOL" in df.columns and "DATE" in df.columns:
                df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
                _NSE_CALENDAR_CACHE = df
                
                # Save clean CSV to local file cache
                try:
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write(csv_text)
                    logger.info(f"Successfully cached clean event calendar locally to {local_path}")
                except Exception as save_err:
                    logger.warning(f"Failed to cache event calendar locally: {save_err}")
                
                return df
        except Exception as parse_err:
            logger.warning(f"Failed to parse online downloaded event calendar CSV: {parse_err}")

    # 2. Fallback to Local Files
    logger.info("Searching for local event calendar files...")
    local_patterns = ["*event-calendar*.csv", "*corporate_events*.csv", "*nse_events*.csv", local_path]
    local_files = []
    for pattern in local_patterns:
        local_files.extend(glob.glob(pattern))
    
    local_files = list(set(local_files))
    
    for filepath in sorted(local_files):
        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                content = f.read()
            cleaned_content = clean_event_calendar_csv(content)
            
            df = pd.read_csv(StringIO(cleaned_content))
            if df.index.name is not None:
                df = df.reset_index()
            df.columns = [c.strip().upper() for c in df.columns]
            
            if "SYMBOL" in df.columns and "DATE" in df.columns:
                df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
                _NSE_CALENDAR_CACHE = df
                logger.info(f"Successfully loaded NSE corporate filings from local file: {filepath}")
                
                # Proactively clean corrupt file on disk
                if cleaned_content != content:
                    try:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(cleaned_content)
                        logger.info(f"Proactively fixed and saved clean version of: {filepath}")
                    except Exception as fix_err:
                        logger.warning(f"Failed to save clean version to {filepath}: {fix_err}")
                        
                return df
        except Exception as e:
            logger.warning(f"Found local event CSV file {filepath} but failed to parse: {e}")

    logger.error("No online or local event calendar available.")
    _NSE_CALENDAR_CACHE = pd.DataFrame()
    return _NSE_CALENDAR_CACHE

def analyze_stock_news(symbol: str) -> dict:
    """
    Fetches the 5 most recent news items from yfinance and calculates:
      - sentiment_score: Net value from -1.0 (very bearish) to +1.0 (very bullish)
      - sentiment_label: 'Bullish', 'Bearish', or 'Neutral'
      - headlines: List of headline titles processed
      - flag_bearish_news: True if critical negative words (fraud, probe) are found
    """
    result = {
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral",
        "headlines": [],
        "flag_bearish_news": False,
        "critical_alert": None
    }
    
    try:
        ticker = yf.Ticker(symbol)
        news_items = ticker.news
        if not news_items:
            return result
            
        total_score = 0.0
        processed_count = 0
        critical_flags = []
        
        # Look at the top 5 articles
        for item in news_items[:5]:
            title = item.get("title", "")
            if not title:
                continue
                
            title_lower = title.lower()
            result["headlines"].append(title)
            
            # Simple keyword scoring
            pos_count = sum(1 for word in BULLISH_KEYWORDS if word in title_lower)
            neg_count = sum(1 for word in BEARISH_KEYWORDS if word in title_lower)
            
            # Extra weight to critical warning words
            critical_warnings = ["probe", "fraud", "penalty", "default", "scam", "investigation"]
            for warning in critical_warnings:
                if warning in title_lower:
                    neg_count += 3
                    critical_flags.append(warning)
            
            score = 0.0
            total_words = pos_count + neg_count
            if total_words > 0:
                score = (pos_count - neg_count) / total_words
                
            total_score += score
            processed_count += 1
            
        if processed_count > 0:
            avg_score = round(total_score / processed_count, 2)
            result["sentiment_score"] = avg_score
            
            # Map score to label
            if avg_score >= 0.15:
                result["sentiment_label"] = "Bullish"
            elif avg_score <= -0.15:
                result["sentiment_label"] = "Bearish"
            else:
                result["sentiment_label"] = "Neutral"
                
        if critical_flags:
            result["flag_bearish_news"] = True
            result["critical_alert"] = f"Negative catalyst detected in news: {', '.join(set(critical_flags))}"
            
    except Exception as e:
        logger.warning(f"Failed to analyze news for {symbol}: {e}")
        
    return result

def check_event_risk(symbol: str, safety_days: int = 5, headlines: list = None) -> dict:
    """
    Checks if there are any upcoming earnings or major events within the safety window.
    Integrates 3 layers of checks:
      1. Official NSE corporate calendar event CSV (cached).
      2. Forward-looking news headline keywords fallback (extremely reliable for Indian stocks).
      3. Yahoo Finance calendar fallback.
    """
    result = {
        "has_event_risk": False,
        "days_to_earnings": None,
        "earnings_date": None,
        "reason": None
    }
    
    today = datetime.now().date()
    max_date = today + timedelta(days=safety_days)
    
    # ─── Layer 1: Cached Official NSE corporate event calendar ──────────────────
    try:
        nse_df = get_nse_corporate_events()
        if not nse_df.empty and "SYMBOL" in nse_df.columns and "DATE" in nse_df.columns:
            clean_sym = symbol.replace(".NS", "").strip()
            stock_events = nse_df[nse_df["SYMBOL"] == clean_sym]
            
            for _, row in stock_events.iterrows():
                event_date_str = str(row["DATE"]).strip()
                # NSE calendar format is typically '%d-%b-%Y' (e.g. 08-Jul-2026)
                try:
                    event_date = pd.to_datetime(event_date_str).date()
                    if today <= event_date <= max_date:
                        result["has_event_risk"] = True
                        result["days_to_earnings"] = (event_date - today).days
                        result["earnings_date"] = event_date
                        purpose = row.get("PURPOSE", "Event")
                        result["reason"] = f"NSE Calendar: {purpose} on {event_date}"
                        return result  # priority match found, exit early
                except Exception:
                    continue
    except Exception as e:
        logger.debug(f"NSE Calendar row matching failed: {e}")

    # ─── Layer 2: Headline-based event scanning (reliability fallback) ──────────
    if headlines:
        for headline in headlines:
            headline_lower = headline.lower()
            # Check for explicit board meetings or Q1-Q4 results reports
            if any(kw in headline_lower for kw in ["board meet", "q1 results", "q2 results", "q3 results", "q4 results", "q1 earnings", "q2 earnings", "q3 earnings", "q4 earnings"]):
                result["has_event_risk"] = True
                result["reason"] = f"Headline Catalyst: '{headline}'"
                result["days_to_earnings"] = 1  # Flag immediate risk
                return result
            elif "results" in headline_lower or "earnings" in headline_lower:
                # Look for forward-looking indicators to avoid flagging ancient historical results reports
                if any(x in headline_lower for x in ["announce", "preview", "date", "ahead", "focus", "expect", "to report", "board to", "consider"]):
                    result["has_event_risk"] = True
                    result["reason"] = f"Headline Catalyst: '{headline}'"
                    result["days_to_earnings"] = 2
                    return result
                    
    # ─── Layer 3: Yahoo Finance Calendar API check (fallback) ───────────────────
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        
        # Calendar can be a dataframe or dict depending on yfinance version
        if calendar is not None:
            dates = None
            if isinstance(calendar, dict) and "Earnings Date" in calendar:
                dates = calendar["Earnings Date"]
            elif isinstance(calendar, pd.DataFrame) and "Earnings Date" in calendar.index:
                dates = calendar.loc["Earnings Date"].values
                
            if dates and len(dates) > 0:
                earnings_date = pd.to_datetime(dates[0]).to_pydatetime()
                time_diff = earnings_date - datetime.now()
                if timedelta(0) <= time_diff <= timedelta(days=safety_days):
                    result["has_event_risk"] = True
                    result["days_to_earnings"] = time_diff.days
                    result["earnings_date"] = earnings_date.date()
                    result["reason"] = f"Calendar Event: Q1/Q2 earnings on {earnings_date.date()}"
                    return result
    except Exception as e:
        logger.debug(f"Calendar event fetch failed for {symbol}: {e}")
        
    return result
