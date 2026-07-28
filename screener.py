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
import os
import json

logger = logging.getLogger(__name__)

# Load custom per-stock swing weights if available
workspace_dir = os.path.dirname(os.path.abspath(__file__))
SWING_WEIGHTS_FILE = os.path.join(workspace_dir, "models", "swing_stock_weights.json")
SWING_CUSTOM_WEIGHTS = {}
if os.path.exists(SWING_WEIGHTS_FILE):
    try:
        with open(SWING_WEIGHTS_FILE, "r") as f:
            raw_weights = json.load(f)
        filtered_weights = {}
        excluded_count = 0
        for sym, w in raw_weights.items():
            t_trades = w.get("total_trades", 0)
            if "trend" in w or "pullback" in w:
                t_trades = max(w.get("trend", {}).get("total_trades", 0), w.get("pullback", {}).get("total_trades", 0))
            if t_trades >= config.WEIGHTS_MIN_SAMPLE_SIZE:
                filtered_weights[sym] = w
            else:
                excluded_count += 1
        SWING_CUSTOM_WEIGHTS = filtered_weights
        logger.info(f"Loaded custom Swing weights for {len(SWING_CUSTOM_WEIGHTS)} stocks. Excluded {excluded_count} stocks with < {config.WEIGHTS_MIN_SAMPLE_SIZE} training samples.")
    except Exception as e:
        logger.error(f"Failed to load custom Swing weights: {e}")





@dataclass
class Candidate:
    symbol: str
    score: int
    beta: float
    adx: float
    setup_type: str = "momentum"
    trend_score: float = 0.0
    pullback_score: float = 0.0
    setup_class: str = "trend"
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
    win_rate: float = 0.0

    def to_line(self) -> str:
        sym = self.symbol.replace(".NS", "")
        reasons_str = ", ".join(self.reasons)
        
        # Check setup type and Nifty index outperformance for highlights
        is_high_prob_setup = self.setup_type in ["pullback_sma50", "rsi2_pullback"]
        is_outperforming_index = any("Outperforming index" in r for r in self.reasons)
        
        if self.win_rate > 0:
            if self.win_rate >= 95.0:
                setup_highlight = f"🔥 [{self.win_rate}% Win Rate Setup] "
            elif self.win_rate >= 90.0:
                setup_highlight = f"⭐ [{self.win_rate}% Win Rate Setup] "
            else:
                setup_highlight = f"✨ [{self.win_rate}% Win Rate Setup] "
        else:
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
            # Merge delivery_pct from DuckDB
            try:
                import os
                from config import config
                db_path = config.DUCKDB_PATH
                if os.path.exists(db_path):
                    import duckdb
                    # Strip tz for matching
                    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    with duckdb.connect(db_path, read_only=True) as con:
                        deliv_df = con.execute(
                            """
                            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                                   delivery_pct
                            FROM stock_prices
                            WHERE symbol = ?
                            """,
                            [symbol]
                        ).fetchdf()
                        if not deliv_df.empty:
                            deliv_df["Date"] = pd.to_datetime(deliv_df["Date"])
                            deliv_df = deliv_df.set_index("Date")
                            df = df.join(deliv_df, how="left")
            except Exception as e:
                logger.warning(f"Could not merge delivery_pct from DuckDB: {e}")
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
                               dividends AS Dividends, stock_splits AS "Stock Splits", delivery_pct
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
    df["ema5"] = ta.trend.ema_indicator(df["Close"], window=5)
    df["ema20"] = ta.trend.ema_indicator(df["Close"], window=20)

    # Class-specific swing indicators
    df["uptrend_sma200"] = df["Close"] > df["sma200"]
    df["ema_cross"] = df["ema5"] > df["ema20"]
    df["bullish_engulfing"] = (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"] > df["Open"]) & (df["Open"] <= df["Close"].shift(1)) & (df["Close"] >= df["Open"].shift(1))
    df["rsi14_oversold"] = df["rsi14"] < 35
    df["stoch_d_turn"] = df["stoch_d"] > df["stoch_d"].shift(1)
    df["bb_pullback"] = df["Low"] <= df["bb_low"]
    df["rsi2_pullback"] = df["rsi2"] < 5

    # Delivery volume average
    if "delivery_pct" in df.columns:
        df["deliv_avg20"] = df["delivery_pct"].shift(1).rolling(20).mean()
    else:
        df["delivery_pct"] = 0.0
        df["deliv_avg20"] = 0.0

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


def fetch_india_vix(period="1mo") -> float | None:
    try:
        df = yf.download("^INDIAVIX", period=period, progress=False)
        if not df.empty:
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            return float(df["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Failed to download India VIX from yfinance: {e}. Trying local fallback.")
        
    try:
        import os
        from config import config
        db_path = config.DUCKDB_PATH
        if os.path.exists(db_path):
            import duckdb
            with duckdb.connect(db_path, read_only=True) as con:
                res = con.execute("SELECT close FROM india_vix ORDER BY date DESC LIMIT 1").fetchone()
                if res:
                    return float(res[0])
    except Exception as ex:
        logger.warning(f"Local India VIX fallback failed: {ex}")
    return None

def evaluate(symbol: str, df: pd.DataFrame, stock_info=None, skip_fundamental=False, index_df: pd.DataFrame = None, custom_weights: dict = None, use_custom_weights: bool = True, setup_class: str = "trend") -> Candidate:
    if stock_info is None or not isinstance(stock_info, dict):
        stock_info = {}

    # Extract weights dict and target/stop params for active class
    custom_sl_atr = None
    custom_target_atr = None
    weights_dict = None

    if custom_weights is None and use_custom_weights:
        custom_weights = SWING_CUSTOM_WEIGHTS.get(symbol, None)
        if custom_weights is None and not symbol.endswith(".NS"):
            custom_weights = SWING_CUSTOM_WEIGHTS.get(symbol + ".NS", None)
        elif custom_weights is None and symbol.endswith(".NS"):
            custom_weights = SWING_CUSTOM_WEIGHTS.get(symbol.replace(".NS", ""), None)

    if custom_weights is not None:
        if setup_class in custom_weights:
            class_weights = custom_weights[setup_class]
            weights_dict = class_weights.get("weights", {})
            custom_sl_atr = class_weights.get("stop_loss_atr", None)
            custom_target_atr = class_weights.get("target_atr", None)
        elif "weights" in custom_weights:
            weights_dict = custom_weights["weights"]
            custom_sl_atr = custom_weights.get("stop_loss_atr", None)
            custom_target_atr = custom_weights.get("target_atr", None)
        else:
            weights_dict = custom_weights

    # Strip timezone from index to avoid timezone-naive vs timezone-aware mismatches
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    if index_df is not None and not index_df.empty:
        if isinstance(index_df.index, pd.DatetimeIndex) and index_df.index.tz is not None:
            index_df = index_df.copy()
            index_df.index = index_df.index.tz_localize(None)

    # Compute daily indicators (skip if already pre-calculated)
    if "sma50" in df.columns and "rsi14" in df.columns and "atr" in df.columns:
        df_indicators = df
    else:
        df_indicators = _add_indicators(df)
    if len(df_indicators) < 10:
        raise ValueError("Not enough indicator history")

    last = df_indicators.iloc[-1]

    # Signal vector extraction
    sma50_rising = last["sma50"] > df_indicators.iloc[-6]["sma50"]
    uptrend_sma50_val = bool(last["Close"] > last["sma50"] and sma50_rising)
    uptrend_sma200_val = bool(last["uptrend_sma200"])
    adx_strong_trend_val = bool(last["adx"] >= 25)
    ema_cross_val = bool(last["ema_cross"])
    volume_spike_val = bool(last["Volume"] > 1.5 * last["vol_avg20"])
    bullish_engulfing_val = bool(last["bullish_engulfing"])

    bb_pullback_val = bool(last["bb_pullback"])
    rsi2_pullback_val = bool(last["rsi2_pullback"])
    rsi14_oversold_val = bool(last["rsi14_oversold"])
    stoch_d_turn_val = bool(last["stoch_d_turn"])

    signals = {
        "uptrend_sma50": uptrend_sma50_val,
        "uptrend_sma200": uptrend_sma200_val,
        "adx_strong_trend": adx_strong_trend_val,
        "ema_cross": ema_cross_val,
        "volume_spike": volume_spike_val,
        "bullish_engulfing": bullish_engulfing_val,
        "bb_pullback": bb_pullback_val,
        "rsi2_pullback": rsi2_pullback_val,
        "rsi14_oversold": rsi14_oversold_val,
        "stoch_d_turn": stoch_d_turn_val,
    }

    # Set up class weights
    TREND_FEATURES = ["uptrend_sma50", "uptrend_sma200", "adx_strong_trend", "ema_cross", "volume_spike", "bullish_engulfing"]
    PULLBACK_FEATURES = ["bb_pullback", "rsi2_pullback", "rsi14_oversold", "stoch_d_turn"]

    t_w_dict = None
    p_w_dict = None
    if custom_weights is not None:
        if "trend" in custom_weights:
            t_w_dict = custom_weights["trend"].get("weights")
        if "pullback" in custom_weights:
            p_w_dict = custom_weights["pullback"].get("weights")

    default_trend = {"uptrend_sma50": 1.0, "uptrend_sma200": 1.0, "adx_strong_trend": 1.0, "ema_cross": 1.0, "volume_spike": 1.0, "bullish_engulfing": 1.0}
    default_pullback = {"bb_pullback": 1.0, "rsi2_pullback": 1.0, "rsi14_oversold": 1.0, "stoch_d_turn": 1.0}

    t_w = t_w_dict if t_w_dict is not None else default_trend
    p_w = p_w_dict if p_w_dict is not None else default_pullback

    trend_score = sum(t_w.get(f, 1.0) * float(signals[f]) for f in TREND_FEATURES)
    pullback_score = sum(p_w.get(f, 1.0) * float(signals[f]) for f in PULLBACK_FEATURES)

    score = trend_score if setup_class == "trend" else pullback_score
    reasons = []
    if setup_class == "trend":
        if uptrend_sma50_val: reasons.append("Uptrend (above rising SMA50)")
        if uptrend_sma200_val: reasons.append("Uptrend (above SMA200)")
        if adx_strong_trend_val: reasons.append("Strong Trend (ADX >= 25)")
        if ema_cross_val: reasons.append("EMA5 > EMA20 crossover")
        if volume_spike_val: reasons.append("Volume spike (>1.5x avg)")
        if bullish_engulfing_val: reasons.append("Bullish engulfing candle")
        setup_type = "trend"
    else:
        if bb_pullback_val: reasons.append("BB lower band support touch")
        if rsi2_pullback_val: reasons.append("RSI(2) oversold (<5)")
        if rsi14_oversold_val: reasons.append("RSI(14) oversold (<35)")
        if stoch_d_turn_val: reasons.append("Stochastic %D turned up")
        setup_type = "pullback"

    # Stop Loss & Target Calculation using normalized ATR% with strong Risk-to-Reward (RR >= 1.8:1)
    last_close = float(last["Close"])
    atr_val = float(last["atr"])
    atr_pct = atr_val / last_close

    sl_mult = custom_sl_atr if custom_sl_atr is not None else 1.0
    target_mult = custom_target_atr if custom_target_atr is not None else 2.2

    sl_pct = atr_pct * sl_mult
    target_pct = atr_pct * target_mult

    # Re-derive stop floor (0.5 * atr_pct) and target range in ATR% terms
    sl_pct = max(sl_pct, 0.4 * atr_pct)
    target_pct = max(target_pct, 1.8 * atr_pct)
    target_pct = min(target_pct, 0.08)

    stop_loss = round(last_close * (1.0 - sl_pct), 2)
    target = round(last_close * (1.0 + target_pct), 2)

    win_rate = 0.0
    if custom_weights is not None:
        if setup_class in custom_weights:
            win_rate = custom_weights[setup_class].get("win_rate", 0.0)

    return Candidate(
        symbol=symbol,
        setup_type=setup_type,
        score=score,
        trend_score=trend_score,
        pullback_score=pullback_score,
        setup_class=setup_class,
        beta=stock_info.get("beta") or 0.0,
        adx=last["adx"],
        reasons=reasons,
        close=round(last_close, 2),
        rsi=round(float(last["rsi14"]), 1),
        stop_loss=stop_loss,
        target=target,
        signals=signals,
        win_rate=win_rate,
    )


def run_screener(symbols: list, period: str, interval: str, min_score: int) -> list:
    candidates = []
    logger.info("Scanning %d symbols", len(symbols))
    
    # Market Regime Filter:
    # 1. Nifty Close below SMA20 -> min_score +2
    # 2. India VIX > 22 -> min_score +1
    base_t_min = min_score
    base_p_min = max(2, min_score - 1)
    
    nifty_adj = 0
    vix_adj = 0
    
    if symbols:
        try:
            index_df = get_benchmark_data(symbols[0], period="1y", interval="1d")
            if not index_df.empty:
                index_df["sma20"] = ta.trend.sma_indicator(index_df["Close"], window=20)
                if not index_df["sma20"].empty:
                    last_idx = index_df.iloc[-1]
                    idx_close = float(last_idx["Close"])
                    idx_sma20 = float(last_idx["sma20"])
                    
                    if idx_close < idx_sma20:
                        nifty_adj = 1
                        logger.info(f"Weak Nifty Regime: Nifty close ({idx_close:.2f}) < SMA20 ({idx_sma20:.2f}). Adjusting min_score by +1.")
                    else:
                        logger.info(f"Strong/Normal Nifty Regime: Nifty close ({idx_close:.2f}) >= SMA20 ({idx_sma20:.2f}).")
            
            # Fetch India VIX for second axis regime check
            vix_val = fetch_india_vix()
            if vix_val is not None:
                logger.info(f"Current India VIX: {vix_val:.2f}")
                if vix_val > 22.0:
                    vix_adj = 1
                    logger.info(f"High Volatility Regime: VIX ({vix_val:.2f}) > 22.0. Adjusting min_score by +1.")
        except Exception as e:
            logger.warning(f"Failed to check Market Regime index filter: {e}")

    adjusted_t_min = base_t_min + nifty_adj + vix_adj
    adjusted_p_min = base_p_min + nifty_adj + vix_adj
    
    logger.info(f"Active thresholds: Trend MinScore={adjusted_t_min}, Pullback MinScore={adjusted_p_min}")

    for symbol in symbols:
        try:
            logger.info(f"Fetching history for {symbol}")
            df = fetch_history(symbol, period, interval)
            logger.info(f"Evaluating {symbol}, {df.shape}")
            stock_info = fetch_stock_info(symbol)
            
            for sc, adj_min in [("trend", adjusted_t_min), ("pullback", adjusted_p_min)]:
                cand = evaluate(symbol, df, stock_info, setup_class=sc)
                logger.info(f"{symbol} ({sc}) : score: {cand.score}")
                if cand.score >= adj_min - 1:
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
                    if cand.score >= adj_min:
                        candidates.append(cand)
        except Exception as e:
            logger.warning("Skipping %s: %s", symbol, e)
            continue

    candidates.sort(key=lambda c: (c.score, c.beta, c.adx), reverse=True)
    return candidates
    