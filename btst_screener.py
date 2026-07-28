import os
import logging
import sys
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from datetime import datetime
import json

# Add the workspace root to sys.path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from config import config
from stock_universe import get_stock_universe
from screener import fetch_history, fetch_stock_info, get_benchmark_data
from news_analyzer import check_event_risk
from notifier import send_telegram, send_whatsapp_twilio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Load custom stock weights if available
MODEL_FILE = os.path.join(workspace_dir, "models", "btst_stock_weights.json")
CUSTOM_WEIGHTS = {}
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "r") as f:
            raw_weights = json.load(f)
        filtered_weights = {}
        excluded_count = 0
        for sym, w in raw_weights.items():
            t_trades = w.get("total_trades", 0)
            if "trend" in w or "pullback" in w:
                t_trades = max(w.get("trend", {}).get("total_trades", 0), w.get("pullback", {}).get("total_trades", 0))
            if t_trades >= 5:
                filtered_weights[sym] = w
            else:
                excluded_count += 1
        CUSTOM_WEIGHTS = filtered_weights
        logger.info(f"Loaded custom BTST parameters for {len(CUSTOM_WEIGHTS)} stocks. Excluded {excluded_count} stocks with < 5 training samples.")
    except Exception as e:
        logger.error(f"Failed to load custom weights from {MODEL_FILE}: {e}")

class BTSTCandidate:
    def __init__(self, symbol, close, change_pct, volume_vs_avg, rsi, stop_loss, target, reasons, win_rate=0.0):
        self.symbol = symbol
        self.close = close
        self.change_pct = change_pct
        self.volume_vs_avg = volume_vs_avg
        self.rsi = rsi
        self.stop_loss = stop_loss
        self.target = target
        self.reasons = reasons
        self.win_rate = win_rate

    def to_line(self) -> str:
        sym = self.symbol.replace(".NS", "")
        reasons_str = ", ".join(self.reasons)
        highlight = ""
        if self.win_rate > 0:
            if self.win_rate >= 95.0:
                highlight = f"🔥 [{self.win_rate}% Win Rate Setup] "
            elif self.win_rate >= 90.0:
                highlight = f"⭐ [{self.win_rate}% Win Rate Setup] "
            else:
                highlight = f"✨ [{self.win_rate}% Win Rate Setup] "
        return (
            f"• {highlight}{sym} [BTST Setup] (RSI: {self.rsi:.1f}, Vol vs Avg: {self.volume_vs_avg:.2f}x)\n"
            f"   CMP: ₹{self.close:.2f} | Today's Return: {self.change_pct:+.2f}%\n"
            f"   SL: ₹{self.stop_loss:.2f} | Target: ₹{self.target:.2f}\n"
            f"   Signals: {reasons_str}"
        )

def evaluate_btst(symbol, df, index_df=None, skip_event_risk=False) -> BTSTCandidate:
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
    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["turnover"] = df["Close"] * df["Volume"]
    df["turnover_avg20"] = df["turnover"].shift(1).rolling(20).mean()
    if "delivery_pct" in df.columns:
        df["deliv_avg20"] = df["delivery_pct"].shift(1).rolling(20).mean()
    df = df.dropna()

    if len(df) < 5:
        raise ValueError("Not enough data after indicator calculation")

    last = df.iloc[-1]
    
    # Load parameters (custom or default)
    params = CUSTOM_WEIGHTS.get(symbol, CUSTOM_WEIGHTS.get(symbol + ".NS" if not symbol.endswith(".NS") else symbol.replace(".NS", ""), {
        "near_high_pct": 0.015,
        "vol_ratio_limit": 1.2,
        "min_return": 1.0,
        "rsi_min": 55,
        "rsi_max": 78,
        "index_filter": "sma50"
    }))
    nh_pct = params.get("near_high_pct", 0.015)
    vr_limit = params.get("vol_ratio_limit", 1.2)
    min_ret = params.get("min_return", 1.0)
    rsi_min = params.get("rsi_min", 55)
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

    # 6. Delivery volume check (ensure absolute delivery volume is higher than average)
    if "delivery_pct" in last and "deliv_avg20" in last and "vol_avg20" in last:
        margin = config.BTST_DELIVERY_MARGIN
        if margin > 0 and last["deliv_avg20"] > 0:
            today_deliv_vol = last["delivery_pct"] * last["Volume"]
            avg_deliv_vol = last["deliv_avg20"] * last["vol_avg20"]
            delivery_ok = today_deliv_vol >= margin * avg_deliv_vol
            if not delivery_ok:
                return None

    # 7. Circuit safety check
    if last["Close"] > 0:
        close_near_high = (last["High"] - last["Close"]) / last["Close"] <= 0.0005
        narrow_range = (last["High"] - last["Low"]) / last["Close"] < 0.005
        if close_near_high and narrow_range:
            logger.info("Excluding %s due to circuit lock: Close ₹%s, High ₹%s, Low ₹%s", symbol, last["Close"], last["High"], last["Low"])
            return None

    # 8. Liquidity safety check (20d avg turnover >= 1 Crore)
    if "turnover_avg20" in last:
        if last["turnover_avg20"] < 10000000:
            logger.info("Excluding %s due to low turnover: 20d Avg Turnover ₹%s < 1 Crore", symbol, round(last["turnover_avg20"], 2))
            return None

    # 9. Corporate events / board meeting exclusion (next 24h)
    if not skip_event_risk:
        try:
            risk_res = check_event_risk(symbol, safety_days=1)
            if risk_res["has_event_risk"]:
                logger.info("Excluding %s due to earnings/event risk: %s", symbol, risk_res["reason"])
                return None
        except Exception as e:
            logger.warning("Event check failed for %s: %s", symbol, e)

    # Optional Index and Relative Strength Filter
    reasons = [
        "Uptrend (above SMA20 & SMA50)",
        f"Close near high ({round((last['High'] - last['Close']) / last['Close'] * 100, 2)}% off high, limit {round(nh_pct*100, 2)}%)",
        f"Volume spike ({round(vol_ratio, 2)}x avg, limit {vr_limit}x)",
        f"Strong green candle (return {round(today_ret, 2)}%, limit {min_ret}%)",
        f"RSI in momentum zone ({round(last['rsi14'], 1)}, range {rsi_min}-{rsi_max})"
    ]
    if "delivery_pct" in last and "deliv_avg20" in last:
        reasons.append(f"Delivery spike ({round(last['delivery_pct'], 1)}% vs 20-day average {round(last['deliv_avg20'], 1)}%)")

    cw_match = CUSTOM_WEIGHTS.get(symbol, CUSTOM_WEIGHTS.get(symbol + ".NS" if not symbol.endswith(".NS") else symbol.replace(".NS", ""), None))
    if cw_match is not None:
        reasons.append("Matched custom stock-specific parameters")

    last_date = df.index[-1]
    
    # Index trend filter
    index_ok = True
    if index_df is not None and not index_df.empty:
        idx_slice = index_df.loc[index_df.index <= last_date]
        if not idx_slice.empty and len(idx_slice) >= 20:
            idx_close = idx_slice["Close"]
            if idx_filter == "sma20":
                idx_sma = ta.trend.sma_indicator(idx_close, window=20)
                if not idx_sma.empty and not pd.isna(idx_sma.iloc[-1]):
                    index_ok = float(idx_close.iloc[-1]) > float(idx_sma.iloc[-1])
                    if index_ok:
                        reasons.append("Broader market Nifty 50 in short-term uptrend (above SMA20)")
            elif idx_filter == "sma50" and len(idx_slice) >= 50:
                idx_sma = ta.trend.sma_indicator(idx_close, window=50)
                if not idx_sma.empty and not pd.isna(idx_sma.iloc[-1]):
                    index_ok = float(idx_close.iloc[-1]) > float(idx_sma.iloc[-1])
                    if index_ok:
                        reasons.append("Broader market Nifty 50 in medium-term uptrend (above SMA50)")
                        
            # 20-day relative strength outperformance comparison
            if len(df) >= 20 and len(idx_slice) >= 20:
                stock_perf = (last["Close"] - df.iloc[-20]["Close"]) / df.iloc[-20]["Close"]
                index_perf = (idx_close.iloc[-1] - idx_slice.iloc[-20]["Close"]) / idx_slice.iloc[-20]["Close"]
                if stock_perf > index_perf:
                    reasons.append("Outperforming index (20-day relative strength)")

    if not index_ok:
        return None

    last_close = float(last["Close"])
    atr_val = float(last["atr"])
    
    # BTST parameters: Target and Stop Loss based on normalized ATR%
    atr_pct = atr_val / last_close
    target_pct = atr_pct * config.BTST_ATR_MULTIPLIER
    stop_loss_pct = atr_pct * config.BTST_ATR_MULTIPLIER
    
    target = round(last_close * (1.0 + target_pct), 2)
    stop_loss = round(last_close * (1.0 - stop_loss_pct), 2)

    win_rate = 0.0
    cw = CUSTOM_WEIGHTS.get(symbol, CUSTOM_WEIGHTS.get(symbol + ".NS" if not symbol.endswith(".NS") else symbol.replace(".NS", ""), None))
    if cw is not None:
        win_rate = cw.get("win_rate", 0.0)

    return BTSTCandidate(
        symbol=symbol,
        close=last_close,
        change_pct=today_ret,
        volume_vs_avg=vol_ratio,
        rsi=float(last["rsi14"]),
        stop_loss=stop_loss,
        target=target,
        reasons=reasons,
        win_rate=win_rate
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
