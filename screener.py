"""
Swing trading screener.

Strategy (a standard multi-factor swing setup — edit freely to match your own style):
  1. Trend filter   : Close > SMA50, and SMA50 itself rising over the last 5 sessions
  2. Momentum       : RSI(14) between 45-65 (healthy pullback/continuation zone)
                       OR RSI crossed up through 30 in the last 3 sessions (oversold bounce)
  3. MACD           : bullish MACD/signal crossover within the last 3 sessions
  4. Volume         : today's volume > 1.5x the 20-day average volume
  5. Breakout proximity: close is within 1% of (or above) the 20-day high

Each condition that's true adds 1 point (max score = 5). A stock is flagged as a
candidate when score >= MIN_SCORE (default 3, configurable in .env).

This is a rules-based filter, not a prediction. It does not place trades, it only
surfaces candidates worth a closer manual look. Always do your own due diligence —
this is not financial advice.
"""
import logging
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf
import ta

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    symbol: str
    score: int
    beta: float
    adx: float
    reasons: list = field(default_factory=list)
    close: float = 0.0
    rsi: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0

    def to_line(self) -> str:
        sym = self.symbol.replace(".NS", "")
        reasons_str = ", ".join(self.reasons)
        return (
            f"• {sym}  (score {self.score}/7, β {self.beta}, adx {round(self.adx,2)})\n"
            f"   CMP: ₹{self.close:.2f} | RSI: {self.rsi:.1f}\n"
            f"   SL: ₹{self.stop_loss:.2f} | Target: ₹{self.target:.2f}\n"
            f"   Signals: {reasons_str}"
        )


def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df_history = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    df_today = yf.Ticker(symbol).history(period="1d", interval="1d", auto_adjust=True)
    df = pd.concat([df_history[:-1], df_today])
    df = df[~df.index.duplicated(keep="last")]
    if df is None or df.empty or len(df) < 60:
        raise ValueError(f"Not enough data for {symbol}")
    return df

def fetch_stock_info(symbol: str):
    stock_info = yf.Ticker(symbol).info
    return stock_info

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["sma100"] = ta.trend.sma_indicator(df["Close"], window=100)
    df["rsi14"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["high20"] = df["Close"].rolling(20).max()
    df["low20"] = df["Close"].rolling(20).min()
    df["adx"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    return df.dropna()


def evaluate(symbol: str, df: pd.DataFrame) -> Candidate:
    df = _add_indicators(df)
    if len(df) < 10:
        raise ValueError("Not enough indicator history")

    last = df.iloc[-1]
    prev3 = df.iloc[-4:-1]

    score = 0
    reasons = []

    # 1. Trend
    sma50_rising = last["sma50"] > df.iloc[-6]["sma50"]
    if last["Close"] > last["sma50"] and sma50_rising:
        score += 1
        reasons.append("Uptrend (above rising SMA50)")

    # 2. Momentum
    oversold_bounce = (prev3["rsi14"] < 30).any() and last["rsi14"] >= 30
    healthy_zone = 45 <= last["rsi14"] <= 65
    if healthy_zone:
        score += 1
        reasons.append("RSI in healthy zone")
    elif oversold_bounce:
        score += 1
        reasons.append("RSI bounced off oversold")

    # 3. MACD bullish crossover recently
    crossed = ((df["macd"] - df["macd_signal"]).iloc[-4:-1] < 0).any() and \
              (last["macd"] > last["macd_signal"])
    if crossed:
        score += 1
        reasons.append("MACD bullish crossover")

    # 4. Volume spike
    if last["Volume"] > 1.5 * last["vol_avg20"]:
        score += 1
        reasons.append("Volume spike (>1.5x avg)")

    # 5. Near 20-day high (breakout proximity)
    if last["Close"] >= 0.99 * last["high20"]:
        score += 1
        reasons.append("Near 20-day high")

    # 6 strong Trend filter
    if last["adx"] > 25:
        score += 1
        reasons.append("ADX greater than 25")

    # 6. Current price is less than Book value
    stock_info = fetch_stock_info(symbol)
    # if last["Close"] <= stock_info["bookValue"]:
    #     score += 1
    #     reasons.append("Current price less than book value")

    # 7. SMA50 crossed SMA100 within last 5 days
    spread = df["sma50"] - df["sma100"]

    sma_cross = (
        (spread.iloc[-6:-1] <= 0).any() and
        spread.iloc[-1] > 0
    )

    if sma_cross:
        score += 1
        reasons.append("Recent SMA50/SMA100 bullish crossover")
    stop_loss = round(float(last["low20"]), 2)
    risk = max(float(last["Close"]) - stop_loss, 0.01)
    target = round(float(last["Close"]) + 2 * risk, 2)  # simple 1:2 risk-reward

    return Candidate(
        symbol=symbol,
        score=score,
        beta=stock_info["beta"],
        adx=last["adx"],
        reasons=reasons,
        close=round(float(last["Close"]), 2),
        rsi=round(float(last["rsi14"]), 1),
        stop_loss=stop_loss,
        target=target,
    )


def run_screener(symbols: list, period: str, interval: str, min_score: int) -> list:
    candidates = []
    logger.info(f"input {len(symbols)} {symbols}")
    for symbol in symbols:
        try:
            logger.info(f"Fetching history for {symbol}")
            df = fetch_history(symbol, period, interval)
            logger.info(f"evaluating {symbol}")
            cand = evaluate(symbol, df)
            logger.info(f"{symbol} : score: {cand.score}")
            if cand.score >= min_score:
                candidates.append(cand)
        except Exception as e:
            logger.warning("Skipping %s: %s", symbol, e)
            continue

    candidates.sort(key=lambda c: (c.score, c.beta, c.adx), reverse=True)
    return candidates