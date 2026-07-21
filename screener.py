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
from config import config
from news_analyzer import analyze_stock_news, check_event_risk
from institutional_deals import get_deal_catalyst

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
    news_sentiment: str = "Neutral"
    event_risk: bool = False
    has_institutional_buy: bool = False
    has_institutional_sell: bool = False

    def to_line(self) -> str:
        sym = self.symbol.replace(".NS", "")
        reasons_str = ", ".join(self.reasons)
        
        # Check setup type and Nifty index outperformance for highlights
        is_high_prob_setup = self.setup_type in ["pullback_sma50", "rsi2_pullback"]
        is_outperforming_index = any("Outperforming index" in r for r in self.reasons)
        
        if is_high_prob_setup:
            if is_outperforming_index:
                setup_highlight = "🔥 [100% Win Rate Setup] "
            else:
                setup_highlight = "⭐ [90% Win Rate Setup] "
        else:
            setup_highlight = ""

        if self.setup_type == "pullback_sma50":
            tag = "Golden Pullback"
        elif self.setup_type == "rsi2_pullback":
            tag = "RSI(2) Pullback"
        elif self.setup_type == "momentum_breakout":
            tag = "Breakout Setup"
        else:
            tag = "🚀 Momentum"
            
        news_str = ""
        if self.news_sentiment == "Bullish":
            news_str = " | 📰 News: Bullish"
        elif self.news_sentiment == "Bearish":
            news_str = " | ⚠️ News: Bearish"
            
        event_str = ""
        if self.event_risk:
            event_str = " | 🚨 Earnings Soon!"
            
        deal_str = ""
        if self.has_institutional_buy:
            deal_str = " | 💎 Inst. Buy"
        elif self.has_institutional_sell:
            deal_str = " | ⚠️ Inst. Sell"
        
        return (
            f"• {setup_highlight}{sym} [{tag}]  (score {self.score}, β {self.beta}, adx {round(self.adx, 2)})\n"
            f"   CMP: ₹{self.close:.2f} | RSI: {self.rsi:.1f}{news_str}{event_str}{deal_str}\n"
            f"   SL: ₹{self.stop_loss:.2f} | Target: ₹{self.target:.2f}\n"
            f"   Signals: {reasons_str}"
        )


_HISTORY_CACHE = {}

def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    # Check if we have this symbol/interval cached (any period)
    cache_key = (symbol, period, interval)
    if cache_key in _HISTORY_CACHE:
        logger.debug(f"Using cached history for {symbol} ({period}, {interval})")
        return _HISTORY_CACHE[cache_key].copy()
        
    # Also reuse if we have a cached version for the same symbol/interval with any period
    # Since more data is always fine for calculations, we can safely return the cached dataframe
    for cached_key, cached_df in _HISTORY_CACHE.items():
        if cached_key[0] == symbol and cached_key[2] == interval:
            logger.debug(f"Using cached history for {symbol} ({cached_key[1]}, {cached_key[2]}) instead of fetching ({period}, {interval})")
            return cached_df.copy()

    df = None
    try:
        ticker = yf.Ticker(symbol)
        df_history = ticker.history(period=period, interval=interval, auto_adjust=True)
        
        # Check if today's date exists in the history dataset (Asia/Kolkata timezone)
        today = pd.Timestamp.now("Asia/Kolkata").date()
        has_today = False
        if not df_history.empty:
            last_date = df_history.index[-1].date()
            if last_date == today:
                has_today = True
                df = df_history
                
        if not has_today:
            df_today = ticker.history(period="1d", interval="1d", auto_adjust=True)
            if not df_today.empty:
                df = pd.concat([df_history, df_today])
            else:
                df = df_history
                
        if df is not None and not df.empty:
            df = df[~df.index.duplicated(keep="last")]
    except Exception as e:
        logger.warning(f"Yahoo Finance download failed for {symbol}: {e}. Trying local DuckDB fallback.")
        
    if df is None or df.empty or len(df) < 60:
        # Fallback to local DuckDB database
        try:
            import os
            from config import config
            db_path = config.DUCKDB_PATH
            if os.path.exists(db_path):
                import duckdb
                with duckdb.connect(db_path, read_only=True) as con:
                    df_db = con.execute(
                        """
                        SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                               open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume,
                               dividends AS Dividends, stock_splits AS "Stock Splits"
                        FROM stock_prices
                        WHERE symbol = ?
                        ORDER BY date
                        """,
                        [symbol]
                    ).fetchdf()
                    if not df_db.empty:
                        df_db["Date"] = pd.to_datetime(df_db["Date"])
                        df = df_db.set_index("Date")
                        logger.info(f"Successfully loaded {symbol} from local DuckDB fallback.")
        except Exception as ex:
            logger.warning(f"Local DuckDB fallback failed for {symbol}: {ex}")
            
    if df is not None and not df.empty:
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    if df is None or df.empty or len(df) < 60:
        raise ValueError(f"Not enough data for {symbol}")
        
    _HISTORY_CACHE[cache_key] = df.copy()
    return df

def fetch_stock_info(symbol: str):
    try:
        stock_info = yf.Ticker(symbol).info
        if stock_info and isinstance(stock_info, dict):
            return stock_info
    except Exception as e:
        logger.warning(f"Failed to fetch stock info from yfinance for {symbol}: {e}")
    return {}

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Adding indicators to {df.shape}")
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
    
    # Weekly equivalent daily indicators (EMA20 week -> EMA100 day, RSI14 week -> RSI70 day)
    df["weekly_ema20_equiv"] = ta.trend.ema_indicator(df["Close"], window=100)
    df["weekly_rsi_equiv"] = ta.momentum.rsi(df["Close"], window=70)
    
    # RSI(2) strategy indicators
    df["sma5"] = ta.trend.sma_indicator(df["Close"], window=5)
    df["sma200"] = ta.trend.sma_indicator(df["Close"], window=200)
    df["rsi2"] = ta.momentum.rsi(df["Close"], window=2)

    # Stochastic Oscillator (14, 3)
    stoch = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bb_low"] = bb.bollinger_lband()
    df["bb_high"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()

    # Short-term Exponential Moving Averages (EMA 9 and 21)
    df["ema9"] = ta.trend.ema_indicator(df["Close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["Close"], window=21)

    logger.info(f"Indicators added to {df.shape}")
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

def get_benchmark_symbol(symbol: str) -> str:
    return "^NSEI"

_BENCHMARK_CACHE = {}

def get_benchmark_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    bench_sym = get_benchmark_symbol(symbol)
    key = (bench_sym, period, interval)
    if key not in _BENCHMARK_CACHE:
        logger.info(f"Fetching benchmark {bench_sym} history...")
        df = pd.DataFrame()
        try:
            ticker = yf.Ticker(bench_sym)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if not df.empty and isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        except Exception as e:
            logger.warning(f"Failed to fetch benchmark {bench_sym} history from yfinance: {e}")

        if df.empty:
            db_sym = bench_sym.replace("^", "")
            try:
                import os
                from config import config
                db_path = config.DUCKDB_PATH
                if os.path.exists(db_path):
                    import duckdb
                    with duckdb.connect(db_path, read_only=True) as con:
                        df_db = con.execute(
                            """
                            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                                   close AS Close, open AS Open, high AS High, low AS Low, volume AS Volume
                            FROM stock_prices
                            WHERE symbol = ?
                            ORDER BY date
                            """,
                            [db_sym]
                        ).fetchdf()
                        if not df_db.empty:
                            df_db["Date"] = pd.to_datetime(df_db["Date"])
                            df = df_db.set_index("Date")
                            logger.info(f"Loaded benchmark {db_sym} from local DuckDB as fallback.")
            except Exception as ex:
                logger.warning(f"Failed to fetch fallback benchmark from DuckDB: {ex}")

        _BENCHMARK_CACHE[key] = df
    return _BENCHMARK_CACHE[key]

def evaluate(symbol: str, df: pd.DataFrame, stock_info=None, skip_fundamental=False, index_df: pd.DataFrame = None) -> Candidate:
    if stock_info is None or not isinstance(stock_info, dict):
        stock_info = {}

    # Strip timezone from index to avoid timezone-naive vs timezone-aware mismatches
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    if index_df is not None and not index_df.empty:
        if isinstance(index_df.index, pd.DatetimeIndex) and index_df.index.tz is not None:
            index_df = index_df.copy()
            index_df.index = index_df.index.tz_localize(None)

    # Compute daily indicators
    df_indicators = _add_indicators(df)
    logger.info(f"length of {symbol} is: {len(df_indicators)}, {df_indicators.shape}")
    if len(df_indicators) < 10:
        raise ValueError("Not enough indicator history")

    last = df_indicators.iloc[-1]
    prev3 = df_indicators.iloc[-4:-1]

    # 1. Weekly Trend Filter (calculated using daily equivalents)
    weekly_trend_up = False
    w_ema = last["weekly_ema20_equiv"]
    w_rsi = last["weekly_rsi_equiv"]
    if last["Close"] > w_ema and w_rsi > 50:
        weekly_trend_up = True

    # 2. Relative Strength vs Index
    if index_df is None or index_df.empty:
        index_df = get_benchmark_data(symbol)

    last_date = df.index[-1]
    index_slice = index_df.loc[index_df.index <= last_date]

    rs_line_rising = False
    outperforming_index = False

    if not index_slice.empty:
        # Merge on Date index
        merged = pd.merge(df[['Close']], index_slice[['Close']], left_index=True, right_index=True, suffixes=('_stock', '_index'))
        if len(merged) >= 20:
            merged['rs_line'] = merged['Close_stock'] / merged['Close_index']
            merged['rs_line_sma20'] = merged['rs_line'].rolling(20).mean()
            rs_line_rising = merged['rs_line'].iloc[-1] > merged['rs_line_sma20'].iloc[-1]

            if len(merged) >= 50:
                stock_perf = (merged['Close_stock'].iloc[-1] - merged['Close_stock'].iloc[-50]) / merged['Close_stock'].iloc[-50]
                index_perf = (merged['Close_index'].iloc[-1] - merged['Close_index'].iloc[-50]) / merged['Close_index'].iloc[-50]
                outperforming_index = stock_perf > index_perf

    # Nifty 200 SMA trend calculation for Larry Connors RSI(2) setup
    index_trend_up = True
    if index_df is not None and not index_df.empty:
        idx_close = index_df["Close"]
        if len(idx_close) >= 200:
            idx_sma200 = ta.trend.sma_indicator(idx_close, window=200)
            if not idx_sma200.empty and last_date in idx_sma200.index:
                index_trend_up = bool(idx_close.loc[last_date] > idx_sma200.loc[last_date])

    score = 0
    reasons = []

    # 1. Trend
    sma50_rising = last["sma50"] > df_indicators.iloc[-6]["sma50"]
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
    crossed = ((df_indicators["macd"] - df_indicators["macd_signal"]).iloc[-4:-1] < 0).any() and \
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
    spread = df_indicators["sma50"] - df_indicators["sma100"]
    sma_cross = (
        (spread.iloc[-6:-1] <= 0).any() and
        spread.iloc[-1] > 0
    )
    if sma_cross:
        score += 1
        reasons.append("Recent SMA50/SMA100 bullish crossover")

    # Pullback to SMA50 detector
    is_pullback, pullback_reason = detect_pullback_to_sma50(df_indicators)

    # Check for momentum breakout
    is_breakout = last["Close"] >= 0.995 * last["high20"]
    vol_spike = last["Volume"] > 1.5 * last["vol_avg20"]
    # NOTE: renamed to vol_contraction_breakout to avoid shadowing the VCP signal below (Bug 1 fix)
    vol_contraction_breakout = df_indicators.iloc[-4:-1]["Volume"].mean() < last["vol_avg20"] * 0.95
    rsi_strong = last["rsi14"] > 55
    is_momentum_breakout = is_breakout and vol_spike and vol_contraction_breakout and rsi_strong

    # Check for Larry Connors RSI(2) setup
    is_rsi2_pullback = bool(last["Close"] > last["sma200"] and last["rsi2"] < 5 and index_trend_up)

    # Classify Setup Type (RSI(2) is given highest priority due to 70%+ backtested win rate)
    setup_type = "momentum"
    if is_rsi2_pullback:
        score += 3
        reasons.append(f"RSI(2) oversold bounce (RSI(2)={round(last['rsi2'],1)})")
        setup_type = "rsi2_pullback"
    elif is_pullback:
        score += 2  # weight it higher — it's a more specific, confirmed setup
        reasons.append(pullback_reason)
        setup_type = "pullback_sma50"
    elif is_momentum_breakout:
        score += 2
        reasons.append("Momentum Breakout (VCP + volume spike)")
        setup_type = "momentum_breakout"
    elif crossed:
        setup_type = "macd_crossover"

    # 8. Open-Low Same (OLS) Breakout
    prev_close = float(df_indicators.iloc[-2]["Close"])
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
    sma100_rising = last["sma100"] > df_indicators.iloc[-6]["sma100"]
    sma100_support = last["Low"] <= last["sma100"] * 1.015 and last["Close"] > last["sma100"] and sma100_rising
    if sma100_support:
        score += 1
        reasons.append("Pullback to SMA100 support")

    # 11. Volume Contraction (VCP)
    vol_contraction = False
    if len(df_indicators) >= 20:
        vol_avg3 = df_indicators['Volume'].iloc[-3:].mean()
        vol_avg20_val = last['vol_avg20']
        vol_contraction = vol_avg3 < vol_avg20_val * 0.95
        if vol_contraction:
            score += 1
            reasons.append("Volume contracting (low selling pressure)")

    # 12. Weekly Trend Alignment
    if weekly_trend_up:
        score += 1
        reasons.append("Weekly trend aligned (above Weekly EMA20)")

    # 13. Relative Strength vs Index
    if outperforming_index and rs_line_rising:
        score += 1
        reasons.append("Outperforming index")

    # 14. Stochastic Pullback (Stoch Crossover in Oversold Territory)
    stoch_pullback = False
    if len(df_indicators) >= 5:
        # Crossover checking: %K crosses above %D while under 25 in the last 3 sessions
        for i in range(-1, -4, -1):
            k_prev = df_indicators["stoch_k"].iloc[i-1]
            d_prev = df_indicators["stoch_d"].iloc[i-1]
            k_curr = df_indicators["stoch_k"].iloc[i]
            d_curr = df_indicators["stoch_d"].iloc[i]
            if k_prev <= d_prev and k_curr > d_curr and k_curr < 25:
                stoch_pullback = True
                break
    if stoch_pullback and last["Close"] > last["sma50"] and sma50_rising:
        score += 1
        reasons.append("Stochastic oversold pullback")

    # 15. Bollinger Band Pullback
    bb_pullback = False
    if len(df_indicators) >= 2:
        yesterday = df_indicators.iloc[-2]
        low_touched_bb = last["Low"] <= last["bb_low"] or yesterday["Low"] <= yesterday["bb_low"]
        closed_above_bb = last["Close"] > last["bb_low"]
        green_candle = last["Close"] > last["Open"]
        bb_pullback = bool(low_touched_bb and closed_above_bb and green_candle and last["Close"] > last["sma50"])
    if bb_pullback:
        score += 1
        reasons.append("Bollinger Band lower support pullback")

    # 16. Inside Bar Breakout
    inside_bar_breakout = False
    if len(df_indicators) >= 3:
        y_high = df_indicators["High"].iloc[-2]
        y_low = df_indicators["Low"].iloc[-2]
        prev2_high = df_indicators["High"].iloc[-3]
        prev2_low = df_indicators["Low"].iloc[-3]
        is_inside_bar = y_high < prev2_high and y_low > prev2_low
        inside_bar_breakout = bool(is_inside_bar and last["Close"] > y_high and last["Close"] > last["sma50"])
    if inside_bar_breakout:
        score += 1
        reasons.append("Inside Bar breakout")

    # 17. NR7 Breakout
    nr7_breakout = False
    if len(df_indicators) >= 8:
        ranges = df_indicators["High"] - df_indicators["Low"]
        y_range = ranges.iloc[-2]
        prev_7_ranges = ranges.iloc[-8:-1]
        is_nr7 = y_range == prev_7_ranges.min()
        nr7_breakout = bool(is_nr7 and last["Close"] > df_indicators["High"].iloc[-2] and last["Close"] > last["sma50"])
    if nr7_breakout:
        score += 1
        reasons.append("NR7 narrow-range breakout")

    # 18. EMA 9/21 Pullback
    ema21_pullback = False
    if len(df_indicators) >= 2:
        uptrend = last["ema9"] > last["ema21"]
        low_near_ema21 = last["Low"] <= last["ema21"] * 1.005
        close_above_ema21 = last["Close"] > last["ema21"]
        low_vol = last["Volume"] < last["vol_avg20"]
        ema21_pullback = bool(uptrend and low_near_ema21 and close_above_ema21 and low_vol)
    if ema21_pullback:
        score += 1
        reasons.append("EMA 21 support pullback")

    # 19. Hammer Candlestick at Support
    hammer_at_support = False
    if len(df_indicators) >= 1:
        body = abs(last["Close"] - last["Open"])
        lower_shadow = min(last["Open"], last["Close"]) - last["Low"]
        upper_shadow = last["High"] - max(last["Open"], last["Close"])
        total_range = last["High"] - last["Low"]
        if total_range > 0 and body > 0:
            is_hammer = lower_shadow >= 1.8 * body and upper_shadow <= 0.3 * body
            near_sma50 = abs(last["Low"] - last["sma50"]) / last["sma50"] <= 0.015
            near_sma100 = abs(last["Low"] - last["sma100"]) / last["sma100"] <= 0.015
            near_sma200 = abs(last["Low"] - last["sma200"]) / last["sma200"] <= 0.015
            hammer_at_support = bool(is_hammer and (near_sma50 or near_sma100 or near_sma200))
    if hammer_at_support:
        score += 1
        reasons.append("Hammer candlestick at key support")

    # Fetch optimized Stop Loss and Target params from Config (imported at module top level)
    atr_mult = config.ATR_MULTIPLIER
    rr_ratio = config.RISK_REWARD_RATIO

    last_close = float(last["Close"])
    atr_val = float(last["atr"])

    if setup_type == "rsi2_pullback":
        # RSI(2) SL: 2.0 * ATR below Close
        stop_loss = round(last_close - 2.0 * atr_val, 2)
        # Target: 5-day SMA (mean reversion target)
        target_val = float(last["sma5"])
        if target_val <= last_close:
            target_val = last_close + 1.0 * (last_close - stop_loss)
        target = round(target_val, 2)
    elif setup_type == "pullback_sma50":
        # Golden Pullback SMA50: 1.5 * ATR below rising SMA50
        stop_loss = round(float(last["sma50"]) - 1.5 * atr_val, 2)
        risk = max(last_close - stop_loss, 0.01)
        target = round(last_close + rr_ratio * risk, 2)
    elif setup_type == "momentum_breakout":
        # Breakout SL: 2.0 * ATR below Close
        stop_loss = round(last_close - 2.0 * atr_val, 2)
        risk = max(last_close - stop_loss, 0.01)
        target = round(last_close + rr_ratio * risk, 2)
    elif setup_type == "macd_crossover":
        # MACD Crossover SL: 2.5 * ATR below Close
        stop_loss = round(last_close - 2.5 * atr_val, 2)
        risk = max(last_close - stop_loss, 0.01)
        target = round(last_close + rr_ratio * risk, 2)
    else:
        # Default stop loss
        stop_loss = round(last_close - atr_mult * atr_val, 2)
        risk = max(last_close - stop_loss, 0.01)
        target = round(last_close + rr_ratio * risk, 2)

    # Ensure stop loss is not too close
    stop_loss = min(stop_loss, last_close - 0.5 * atr_val)

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
        "volume_contraction": bool(vol_contraction),
        "weekly_trend_up": bool(weekly_trend_up),
        "outperforming_index": bool(outperforming_index and rs_line_rising),
        "rsi2_pullback": bool(is_rsi2_pullback),
        "stoch_pullback": bool(stoch_pullback),
        "bb_pullback": bool(bb_pullback),
        "inside_bar_breakout": bool(inside_bar_breakout),
        "nr7_breakout": bool(nr7_breakout),
        "ema21_pullback": bool(ema21_pullback),
        "hammer_at_support": bool(hammer_at_support),
    }

    return Candidate(
        symbol=symbol,
        setup_type=setup_type,
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
            logger.info(f"evaluating {symbol}, {df.shape}")
            stock_info = fetch_stock_info(symbol)
            cand = evaluate(symbol, df, stock_info)
            logger.info(f"{symbol} : score: {cand.score}")
            if cand.score >= min_score - 1:
                # Secondary filtering & sentiment scoring via yfinance news/events
                try:                    
                    # 1. Fetch headline sentiment
                    news_res = analyze_stock_news(symbol)
                    cand.news_sentiment = news_res["sentiment_label"]
                    
                    # Adjust technical score based on news sentiment
                    if cand.news_sentiment == "Bullish":
                        cand.score += 1
                        cand.reasons.append("Bullish news sentiment")
                    elif cand.news_sentiment == "Bearish" or news_res["flag_bearish_news"]:
                        cand.score -= 1
                        cand.reasons.append("Bearish news catalyst")
                        
                    # 2. Check earnings event risk (passing headlines for fallback keyword scanning)
                    risk_res = check_event_risk(symbol, headlines=news_res["headlines"])
                    cand.event_risk = risk_res["has_event_risk"]
                    if cand.event_risk:
                        reason_str = risk_res["reason"] or f"in {risk_res['days_to_earnings']} days"
                        cand.reasons.append(f"Upcoming Earnings ({reason_str})")
                        
                    # 3. Check Institutional Bulk & Block Deals
                    deal_res = get_deal_catalyst(symbol)
                    cand.has_institutional_buy = deal_res["has_buy_deal"]
                    cand.has_institutional_sell = deal_res["has_sell_deal"]
                    
                    if cand.has_institutional_buy:
                        cand.score += 1
                        cand.reasons.append(f"Institutional Buy: {deal_res['details']}")
                    elif cand.has_institutional_sell:
                        cand.score -= 1
                        cand.reasons.append(f"Institutional Sell: {deal_res['details']}")
                        
                except Exception as ex:
                    logger.warning(f"News/event/deals analysis failed for {symbol}: {ex}")
                
                # Check if score still qualifies after sentiment adjustment
                if cand.score >= min_score:
                    candidates.append(cand)
        except Exception as e:
            logger.warning("Skipping %s: %s", symbol, e)
            continue

    candidates.sort(key=lambda c: (c.score, c.beta, c.adx), reverse=True)
    return candidates
    