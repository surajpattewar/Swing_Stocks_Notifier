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

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    symbol: str
    score: int
    beta: float
    reasons: list = field(default_factory=list)

    def to_line(self) -> str:
        sym = self.symbol.replace(".NS", "")
        reasons_str = ", ".join(self.reasons)
        return (
            f"• {sym}  (score {self.score}/7, β {self.beta})\n"
            f"   Signals: {reasons_str}"
        )

def fetch_stock_info(symbol: str):
    stock_info = yf.Ticker(symbol).info
    return stock_info

def evaluate(symbol: str) -> Candidate:
    score = 0
    reasons = []

    # 1. Current price is less than Book value
    stock_info = fetch_stock_info(symbol)
    if stock_info["previousClose"] <= stock_info["bookValue"] or stock_info["dayLow"] <= stock_info["bookValue"]:
        score += 3
        reasons.append("Current price less than book value")

    return Candidate(
        symbol=symbol,
        score=score,
        beta=stock_info["beta"],
        reasons=reasons
    )


def run_screener(symbols: list, period: str, interval: str, min_score: int) -> list:
    candidates = []
    logger.info(f"input {len(symbols)} {symbols}")
    for symbol in symbols:
        try:
            logger.info(f"evaluating {symbol}")
            cand = evaluate(symbol)
            logger.info(f"{symbol} : score: {cand.score}")
            if cand.score >= min_score:
                candidates.append(cand)
        except Exception as e:
            logger.warning("Skipping %s: %s", symbol, e)
            continue

    candidates.sort(key=lambda c: (c.score, c.beta), reverse=True)
    return candidates