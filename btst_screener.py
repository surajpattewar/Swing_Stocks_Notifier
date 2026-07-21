import os
import logging
import sys
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from datetime import datetime

# Add the workspace root to sys.path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from config import config
from stock_universe import get_stock_universe
from screener import fetch_history, fetch_stock_info, get_benchmark_data
from notifier import send_telegram, send_whatsapp_twilio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

import json

# Load custom stock weights if available
MODEL_FILE = os.path.join(workspace_dir, "models", "btst_stock_weights.json")
CUSTOM_WEIGHTS = {}
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "r") as f:
            CUSTOM_WEIGHTS = json.load(f)
        logger.info(f"Loaded custom BTST parameters for {len(CUSTOM_WEIGHTS)} stocks.")
    except Exception as e:
        logger.error(f"Failed to load custom weights from {MODEL_FILE}: {e}")

class BTSTCandidate:
    def __init__(self, symbol, close, change_pct, volume_vs_avg, rsi, stop_loss, target, reasons):
        self.symbol = symbol
        self.close = close
        self.change_pct = change_pct
        self.volume_vs_avg = volume_vs_avg
        self.rsi = rsi
        self.stop_loss = stop_loss
        self.target = target
        self.reasons = reasons

    def to_line(self) -> str:
        sym = self.symbol.replace(".NS", "")
        reasons_str = ", ".join(self.reasons)
        return (
            f"• {sym} [BTST Setup] (RSI: {self.rsi:.1f}, Vol vs Avg: {self.volume_vs_avg:.2f}x)\n"
            f"   CMP: ₹{self.close:.2f} | Today's Return: {self.change_pct:+.2f}%\n"
            f"   SL: ₹{self.stop_loss:.2f} | Target: ₹{self.target:.2f}\n"
            f"   Signals: {reasons_str}"
        )

def evaluate_btst(symbol, df, index_df=None) -> BTSTCandidate:
    # Ensure index is timezone-naive
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    if index_df is not None and not index_df.empty:
        if isinstance(index_df.index, pd.DatetimeIndex) and index_df.index.tz is not None:
            index_df = index_df.copy()
            index_df.index = index_df.index.tz_localize(None)

    if len(df) < 60:
        raise ValueError("Not enough history")

    # Add indicators
    df = df.copy()
    df["sma20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["vol_avg20"] = df["Volume"].shift(1).rolling(20).mean()
    df["rsi14"] = ta.momentum.rsi(df["Close"], window=14)
    df = df.dropna()

    if len(df) < 5:
        raise ValueError("Not enough data after indicator calculation")

    last = df.iloc[-1]
    
    # Load parameters (custom or default)
    params = CUSTOM_WEIGHTS.get(symbol, {
        "near_high_pct": 0.002,
        "vol_ratio_limit": 1.5,
        "min_return": 2.0,
        "rsi_min": 60,
        "rsi_max": 78,
        "index_filter": "sma50"
    })
    nh_pct = params.get("near_high_pct", 0.002)
    vr_limit = params.get("vol_ratio_limit", 1.5)
    min_ret = params.get("min_return", 2.0)
    rsi_min = params.get("rsi_min", 60)
    rsi_max = params.get("rsi_max", 78)
    idx_filter = params.get("index_filter", "sma50")
    
    # 1. Price is above 20 & 50 SMAs (uptrend)
    uptrend = last["Close"] > last["sma20"] and last["Close"] > last["sma50"]
    if not uptrend:
        return None

    # 2. Closes near the high of the day (within nh_pct)
    near_high = last["Close"] >= (1.0 - nh_pct) * last["High"]
    if not near_high:
        return None

    # 3. Volume spike (at least vr_limit of 20-day average)
    vol_ratio = last["Volume"] / last["vol_avg20"]
    vol_spike = vol_ratio >= vr_limit
    if not vol_spike:
        return None

    # 4. Today is a strong green candle (Close > Open and return >= min_ret%)
    today_ret = (last["Close"] - last["Open"]) / last["Open"] * 100
    strong_candle = last["Close"] > last["Open"] and today_ret >= min_ret
    if not strong_candle:
        return None

    # 5. RSI in momentum zone (between rsi_min and rsi_max)
    rsi_momentum = rsi_min <= last["rsi14"] <= rsi_max
    if not rsi_momentum:
        return None

    # Optional Index and Relative Strength Filter
    reasons = [
        "Uptrend (above SMA20 & SMA50)",
        f"Close near high ({round((last['High'] - last['Close']) / last['Close'] * 100, 2)}% off high, limit {round(nh_pct*100, 2)}%)",
        f"Volume spike ({round(vol_ratio, 2)}x avg, limit {vr_limit}x)",
        f"Strong green candle (return {round(today_ret, 2)}%, limit {min_ret}%)",
        f"RSI in momentum zone ({round(last['rsi14'], 1)}, range {rsi_min}-{rsi_max})"
    ]
    if symbol in CUSTOM_WEIGHTS:
        reasons.append("Matched custom stock-specific parameters")

    last_date = df.index[-1]
    
    # Index trend filter
    index_ok = True
    if idx_filter == "sma50" and index_df is not None and not index_df.empty:
        idx_slice = index_df.loc[index_df.index <= last_date]
        if not idx_slice.empty and len(idx_slice) >= 50:
            idx_close = idx_slice["Close"]
            idx_sma50 = ta.trend.sma_indicator(idx_close, window=50)
            if not idx_sma50.empty and last_date in idx_sma50.index:
                index_ok = idx_close.loc[last_date] > idx_sma50.loc[last_date]
                if index_ok:
                    reasons.append("Broader market Nifty 50 in uptrend")
                    
            # Relative Strength outperforming Nifty
            if len(df) >= 20 and len(idx_slice) >= 20:
                stock_perf = (last["Close"] - df.iloc[-20]["Close"]) / df.iloc[-20]["Close"]
                index_perf = (idx_close.loc[last_date] - idx_slice.iloc[-20]["Close"]) / idx_slice.iloc[-20]["Close"]
                if stock_perf > index_perf:
                    reasons.append("Outperforming index (20-day relative strength)")

    if not index_ok:
        return None

    last_close = float(last["Close"])
    
    # BTST parameters: Target +1.5%, Stop Loss -1.5%
    target = round(last_close * 1.015, 2)
    stop_loss = round(last_close * 0.985, 2)

    return BTSTCandidate(
        symbol=symbol,
        close=last_close,
        change_pct=today_ret,
        volume_vs_avg=vol_ratio,
        rsi=float(last["rsi14"]),
        stop_loss=stop_loss,
        target=target,
        reasons=reasons
    )

def run_btst_screener(send_alerts=False):
    logger.info("Starting daily BTST screener...")
    
    # Get 200 stock universe
    symbols = get_stock_universe(max_stocks=200, no_of_stocks=200)
    logger.info("Scanning %d symbols for BTST...", len(symbols))
    
    # Load Nifty index data as benchmark
    index_df = None
    try:
        index_df = get_benchmark_data("^NSEI")
    except Exception as e:
        logger.warning(f"Could not load ^GSPC/^NSEI benchmark: {e}")
        
    candidates = []
    
    for symbol in symbols:
        try:
            df = fetch_history(symbol, config.HISTORY_PERIOD, config.HISTORY_INTERVAL)
            cand = evaluate_btst(symbol, df, index_df=index_df)
            if cand:
                candidates.append(cand)
                logger.info(f"BTST Candidate Found: {symbol}")
        except Exception as e:
            continue
            
    logger.info("Found %d BTST stocks today.", len(candidates))
    
    if not candidates:
        message = f"BTST Screener\n\nNo qualifying BTST setups found today {datetime.today().date()}."
    else:
        lines = [f"BTST Screener — {len(candidates)} candidate(s) found on {datetime.today().date()} \nBy Suraj Pattewar\n"]
        for cand in candidates:
            lines.append(cand.to_line())
        lines.append("\nAuto-generated results, not financial advice.")
        message = "\n".join(lines)
        
    print("\n" + message + "\n")
    
    # Send alerts if requested
    if send_alerts and config.SEND_TELEGRAM:
        sent = send_telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, message)
        logger.info(f"Telegram sent: {sent}")
        
    return candidates

if __name__ == "__main__":
    run_btst_screener(send_alerts=True)
