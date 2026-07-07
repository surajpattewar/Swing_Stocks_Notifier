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

def get_nse_corporate_events() -> pd.DataFrame:
    """
    Fetches the full corporate filings event calendar from NSE India.
    Supports reading from a local CSV file fallback (e.g. event-calendar.csv)
    to bypass Cloudflare blocks/firewalls when running offline.
    """
    global _NSE_CALENDAR_CACHE
    if _NSE_CALENDAR_CACHE is not None:
        return _NSE_CALENDAR_CACHE
        
    import os
    import glob
    
    # ─── Layer 1: Check local CSV files in the workspace ─────────────────────
    local_patterns = ["*event-calendar*.csv", "*corporate_events*.csv", "*nse_events*.csv", "event-calendar.csv"]
    local_files = []
    for pattern in local_patterns:
        local_files.extend(glob.glob(pattern))
        
    # Also check parent directory or subdirectories just in case
    local_files = list(set(local_files))  # remove duplicates
    
    for filepath in sorted(local_files):
        try:
            df = pd.read_csv(filepath)
            # If the index has a name (e.g. SYMBOL), reset it so it becomes a column
            if df.index.name is not None:
                df = df.reset_index()
            # Clean and normalize columns to uppercase
            df.columns = [c.strip().upper() for c in df.columns]
            
            if "SYMBOL" in df.columns and "DATE" in df.columns:
                df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
                _NSE_CALENDAR_CACHE = df
                logger.info(f"Successfully loaded NSE corporate filings from local file: {filepath}")
                return df
        except Exception as e:
            logger.warning(f"Found event CSV file {filepath} but failed to parse: {e}")

    # ─── Layer 2: Online API fetch fallback ──────────────────────────────────
    url = "https://www.nseindia.com/api/event-calendar?index=equities&csv=true"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar",
        "Connection": "keep-alive"
    }
    
    session = requests.Session()
    try:
        # Visit NSE India homepage first to establish session cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = session.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        # Read the CSV response text
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        
        # Strip and normalize columns to uppercase
        df.columns = [c.strip().upper() for c in df.columns]
        if "SYMBOL" in df.columns:
            df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
            
        _NSE_CALENDAR_CACHE = df
        logger.info("Successfully fetched and cached NSE Corporate filings calendar online.")
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch corporate filings calendar from NSE India: {e}. Fallbacks will be used.")
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
