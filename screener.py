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
    setup_type: str = "momentum"
    reasons: list = field(default_factory=list)
    close: float = 0.0
    rsi: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    signals: dict = field(default_factory=dict)

    # def to_line(self) -> str:
    #     sym = self.symbol.replace(".NS", "")
    #     reasons_str = ", ".join(self.reasons)
    #     return (
    #         f"• {sym}  (score {self.score}/7, β {self.beta}, adx {round(self.adx,2)})\n"
    #         f"   CMP: ₹{self.close:.2f} | RSI: {self.rsi:.1f}\n"
    #         f"   SL: ₹{self.stop_loss:.2f} | Target: ₹{self.target:.2f}\n"
    #         f"   Signals: {reasons_str}"
    #     )

    def to_line(self) -> str:
        sym = self.symbol.replace(".NS", "")
        reasons_str = ", ".join(self.reasons)
        tag = "Pullback" if self.setup_type == "pullback_sma50" else "🚀 Momentum"
        return (
            f"• {sym} [{tag}]  (score {self.score}, β {self.beta}, adx {round(self.adx, 2)})\n"
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
    df["vol_avg20"] = df["Volume"].shift(1).rolling(20).mean()
    df["high20"] = df["Close"].shift(1).rolling(20).max()
    df["low20"] = df["Close"].rolling(20).min()
    df["adx"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    atr = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"],
                                         window=14,
                                         )
    df["atr"] = atr.average_true_range()
    return df.dropna()

def _sessions_since_crossover(spread: pd.Series, lookback: int) -> int | None:
    """
    spread = sma50 - sma100. Returns how many sessions ago spread crossed
    from <=0 to >0, if that happened within `lookback` sessions. Else None.
    """
    window = spread.iloc[-(lookback + 1):]
    for i in range(len(window) - 1, 0, -1):
        if window.iloc[i - 1] <= 0 < window.iloc[i]:
            return len(window) - 1 - i
    return None


def detect_pullback_to_sma50(df: pd.DataFrame, cross_lookback: int = 20,
                              touch_tolerance_pct: float = 0.015) -> tuple[bool, str]:
    """
    Golden-cross pullback entry:
      1. SMA50 crossed above SMA100 within the last `cross_lookback` sessions
      2. SMA50 is still rising (trend intact)
      3. Price has pulled back to within touch_tolerance_pct of SMA50 (low touched it)
      4. Today shows a bounce: close > SMA50, close > open, RSI ticking up
    """
    spread = df["sma50"] - df["sma100"]
    cross_ago = _sessions_since_crossover(spread, cross_lookback)
    if cross_ago is None:
        return False, ""

    last = df.iloc[-1]
    prev = df.iloc[-2]

    sma50_rising = last["sma50"] > df.iloc[-6]["sma50"]
    if not sma50_rising:
        return False, ""

    touched_sma50 = last["Low"] <= last["sma50"] * (1 + touch_tolerance_pct)
    bounced = last["Close"] > last["sma50"] and last["Close"] > last["Open"]
    rsi_turning_up = last["rsi14"] > prev["rsi14"]

    if touched_sma50 and bounced and rsi_turning_up:
        return True, f"Pullback to SMA50 ({cross_ago}d after golden cross)"
    return False, ""

def evaluate(symbol: str, df: pd.DataFrame, stock_info) -> Candidate:
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

    # 7. SMA50 crossed SMA100 within last 5 days
    spread = df["sma50"] - df["sma100"]

    sma_cross = (
        (spread.iloc[-6:-1] <= 0).any() and
        spread.iloc[-1] > 0
    )

    if sma_cross:
        score += 1
        reasons.append("Recent SMA50/SMA100 bullish crossover")

    # Pullback findout
    is_pullback, pullback_reason = detect_pullback_to_sma50(df)
    setup_type = "momentum"
    if is_pullback:
        score += 2  # weight it higher — it's a more specific, confirmed setup
        reasons.append(pullback_reason)
        setup_type = "pullback_sma50"

    # 8. Open-Low Same (OLS) Breakout
    prev_close = float(df.iloc[-2]["Close"])
    ols_breakout = last["Close"] > last["Open"] and last["Close"] > prev_close and (last["Open"] - last["Low"]) / last["Open"] <= 0.002
    if ols_breakout:
        score += 1
        reasons.append("Open-Low Same (conviction buy)")

    # 9. Strong RSI Momentum Zone
    strong_rsi = 65 < last["rsi14"] <= 80
    if strong_rsi:
        score += 1
        reasons.append("Strong RSI momentum")

    # 10. Bounce off rising SMA100 support
    sma100_rising = last["sma100"] > df.iloc[-6]["sma100"]
    sma100_support = last["Low"] <= last["sma100"] * 1.015 and last["Close"] > last["sma100"] and sma100_rising
    if sma100_support:
        score += 1
        reasons.append("Pullback to SMA100 support")

    last_close = float(last["Close"])
    if setup_type == "pullback_sma50":
        # structural stop: just under the SMA50 itself (the level being defended),
        # with a small ATR buffer for noise
        stop_loss = round(min(float(last["Low"]), float(last["sma50"])) - 0.5 * float(last["atr"]), 2)
    else:
        # generic volatility stop for momentum/breakout setups
        stop_loss = round(last_close - 1.5 * float(last["atr"]), 2)

    risk = max(last_close - stop_loss, 0.01)
    target = round(last_close + 2 * risk, 2)

    signals = {
        "uptrend_sma50": bool(last["Close"] > last["sma50"] and sma50_rising),
        "rsi_momentum": bool(healthy_zone or oversold_bounce),
        "macd_crossover": bool(crossed),
        "volume_spike": bool(last["Volume"] > 1.5 * last["vol_avg20"]),
        "breakout_proximity": bool(last["Close"] >= 0.99 * last["high20"]),
        "adx_strong_trend": bool(last["adx"] > 25),
        "sma_cross": bool(sma_cross),
        "pullback_sma50": bool(is_pullback),
        "ols_breakout": bool(ols_breakout),
        "strong_rsi": bool(strong_rsi),
        "sma100_support": bool(sma100_support),
    }

    return Candidate(
        symbol=symbol,
        setup_type = setup_type,
        score=score,
        beta=stock_info.get("beta") or 0.0,
        adx=last["adx"],
        reasons=reasons,
        close=round(float(last["Close"]), 2),
        rsi=round(float(last["rsi14"]), 1),
        stop_loss=stop_loss,
        target=target,
        signals=signals,
    )


def run_screener(symbols: list, period: str, interval: str, min_score: int) -> list:
    candidates = []
    logger.info(f"input {len(symbols)} {symbols}")
    for symbol in symbols:
        try:
            logger.info(f"Fetching history for {symbol}")
            df = fetch_history(symbol, period, interval)
            logger.info(f"evaluating {symbol}")
            stock_info = fetch_stock_info(symbol)
            cand = evaluate(symbol, df, stock_info)
            logger.info(f"{symbol} : score: {cand.score}")
            if cand.score >= min_score:
                candidates.append(cand)
        except Exception as e:
            logger.warning("Skipping %s: %s", symbol, e)
            continue

    candidates.sort(key=lambda c: (c.score, c.beta, c.adx), reverse=True)
    return candidates