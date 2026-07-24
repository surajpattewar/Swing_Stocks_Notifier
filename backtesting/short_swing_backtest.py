
# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
"""
short_swing_backtest.py
=======================
Standalone walk-forward backtest for the proposed 'short_swing' setup.

Strategy:
  A stock fires a SHORT-SWING signal when ALL (or 4/5) of these are true:
    1. Uptrend:       Close > SMA50, and SMA50 is rising (today > 5 sessions ago)
    2. Strong ADX:    ADX(14) > 30  (strong directional move, not ranging)
    3. RSI momentum:  RSI(14) in 55-70 zone (bullish but not overbought)
    4. Near breakout: Close >= 99.5% of the 20-day high (within 0.5% of recent top)
    5. Volume surge:  Today's volume > 2x the 20-day rolling average

Entry:  NEXT day's open (realistic: screener runs after close, enter next morning)
Target: +3% above entry price
SL:     -1.5% below entry price  (2:1 R:R)
Hold:   Max 5 trading days before timeout at close

Run:
    uv run python short_swing_backtest.py
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import ta
import yfinance as yf

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
BACKTEST_MONTHS  = 4
MAX_HOLDING_DAYS = 5
TARGET_PCT       = 3.0
STOP_PCT         = 1.5
MAX_WORKERS      = 8
HISTORY_PERIOD   = "12mo"
HISTORY_INTERVAL = "1d"
MIN_CONDITIONS   = 4   # lower to 4 to get more signals, set to 5 for strict

# NSE 100 universe
SYMBOLS = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS",
    "BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS","BEL.NS","BHARTIARTL.NS",
    "CIPLA.NS","COALINDIA.NS","DRREDDY.NS","EICHERMOT.NS","ETERNAL.NS",
    "GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS","HINDALCO.NS",
    "HINDUNILVR.NS","ICICIBANK.NS","INDIGO.NS","INFY.NS","ITC.NS",
    "JIOFIN.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS",
    "MARUTI.NS","MAXHEALTH.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS",
    "POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SHRIRAMFIN.NS","SBIN.NS",
    "SUNPHARMA.NS","TCS.NS","TATACONSUM.NS","TATASTEEL.NS",
    "TECHM.NS","TITAN.NS","TRENT.NS","ULTRACEMCO.NS","WIPRO.NS",
    "ABB.NS","ADANIGREEN.NS","AMBUJACEM.NS","ATGL.NS","AUBANK.NS",
    "BANKBARODA.NS","BHEL.NS","BPCL.NS","CANBK.NS","CHOLAFIN.NS",
    "CUMMINSIND.NS","DMART.NS","DIVISLAB.NS","DLF.NS","GAIL.NS",
    "GODREJCP.NS","HAVELLS.NS","HEROMOTOCO.NS","HINDPETRO.NS","HYUNDAI.NS",
    "ICICIPRULI.NS","IDBI.NS","IOC.NS","IRFC.NS","JINDALSTEL.NS",
    "JSWENERGY.NS","LICI.NS","LODHA.NS","LTM.NS","LUPIN.NS",
    "MARICO.NS","MOTHERSON.NS","MUTHOOTFIN.NS","NAUKRI.NS","NHPC.NS",
    "NMDC.NS","NYKAA.NS","OFSS.NS","PFC.NS","PIDILITIND.NS",
    "PNB.NS","RECLTD.NS","SIEMENS.NS","SRF.NS","TATAPOWER.NS",
    "TORNTPHARM.NS","UNIONBANK.NS","UNITDSPR.NS","VBL.NS",
    "VEDL.NS","ZOMATO.NS","ZYDUSLIFE.NS",
]


# ─── Indicators ───────────────────────────────────────────────────────────────
def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma50"]     = ta.trend.sma_indicator(df["Close"], window=50)
    df["rsi14"]     = ta.momentum.rsi(df["Close"], window=14)
    df["adx"]       = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["vol_avg20"] = df["Volume"].shift(1).rolling(20).mean()
    df["high20"]    = df["Close"].shift(1).rolling(20).max()
    return df.dropna()


# ─── Signal detection ─────────────────────────────────────────────────────────
def _detect_short_swing(df_ind: pd.DataFrame) -> dict | None:
    if len(df_ind) < 6:
        return None
    last = df_ind.iloc[-1]
    sma50_rising = last["sma50"] > df_ind.iloc[-6]["sma50"]

    c1 = bool(last["Close"] > last["sma50"] and sma50_rising)   # Uptrend
    c2 = bool(last["adx"] > 30)                                  # Strong trend
    c3 = bool(55 <= last["rsi14"] <= 70)                         # RSI momentum zone
    c4 = bool(last["Close"] >= 0.995 * last["high20"])           # Near 20-day high
    c5 = bool(last["Volume"] > 2.0 * last["vol_avg20"])          # Volume surge 2x

    return {
        "c1_uptrend": c1, "c2_adx_strong": c2, "c3_rsi_zone": c3,
        "c4_near_high": c4, "c5_vol_surge": c5,
        "conditions_met": sum([c1, c2, c3, c4, c5]),
        "close": float(last["Close"]),
        "rsi": round(float(last["rsi14"]), 1),
        "adx": round(float(last["adx"]), 1),
    }


# ─── Outcome simulation ───────────────────────────────────────────────────────
def _simulate_outcome(df: pd.DataFrame, entry_pos: int, entry_price: float,
                      target: float, stop_loss: float) -> dict:
    """
    Uses next trading days' High/Low after entry_pos to determine if target
    or stop fires first. Conservative: stop wins same-day ties.
    """
    future = df.iloc[entry_pos + 1: entry_pos + 1 + MAX_HOLDING_DAYS]
    if future.empty:
        return {"outcome": "no_data", "exit_price": None,
                "exit_date": None, "days_held": 0, "return_pct": None}

    for i, (dt, row) in enumerate(future.iterrows(), start=1):
        if row["Low"] <= stop_loss:
            return {"outcome": "stop_loss_hit", "exit_price": stop_loss,
                    "exit_date": dt.date(), "days_held": i,
                    "return_pct": round((stop_loss - entry_price) / entry_price * 100, 2)}
        if row["High"] >= target:
            return {"outcome": "target_hit", "exit_price": target,
                    "exit_date": dt.date(), "days_held": i,
                    "return_pct": round((target - entry_price) / entry_price * 100, 2)}

    exit_price = float(future.iloc[-1]["Close"])
    return {"outcome": "timeout", "exit_price": exit_price,
            "exit_date": future.index[-1].date(), "days_held": len(future),
            "return_pct": round((exit_price - entry_price) / entry_price * 100, 2)}


# ─── Per-symbol walk-forward ──────────────────────────────────────────────────
def _run_symbol(symbol: str, backtest_start: date) -> list:
    try:
        df = yf.Ticker(symbol).history(
            period=HISTORY_PERIOD, interval=HISTORY_INTERVAL, auto_adjust=True)
        if df.empty or len(df) < 80:
            return []
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", symbol, e)
        return []

    df_ind       = _compute_indicators(df)
    bt_start     = pd.Timestamp(backtest_start)
    eval_dates   = df_ind.index[df_ind.index >= bt_start]
    results      = []

    for signal_date in eval_dates:
        df_slice = df_ind.loc[:signal_date]
        if len(df_slice) < 60:
            continue

        sig = _detect_short_swing(df_slice)
        if sig is None or sig["conditions_met"] < MIN_CONDITIONS:
            continue

        # Entry = next trading day's open (no look-ahead)
        future_raw = df.loc[df.index > signal_date]
        if future_raw.empty:
            continue
        next_open = float(future_raw.iloc[0]["Open"])

        target    = round(next_open * (1 + TARGET_PCT / 100), 2)
        stop_loss = round(next_open * (1 - STOP_PCT  / 100), 2)

        # positional index for simulation
        entry_pos = df.index.get_loc(future_raw.index[0])
        outcome   = _simulate_outcome(df, entry_pos, next_open, target, stop_loss)

        results.append({
            "symbol":          symbol,
            "signal_date":     signal_date.date(),
            "entry_date":      future_raw.index[0].date(),
            "signal_close":    round(sig["close"], 2),
            "entry_price":     round(next_open, 2),
            "target":          target,
            "stop_loss":       stop_loss,
            "rsi":             sig["rsi"],
            "adx":             sig["adx"],
            "conditions_met":  sig["conditions_met"],
            "c1_uptrend":      sig["c1_uptrend"],
            "c2_adx_strong":   sig["c2_adx_strong"],
            "c3_rsi_zone":     sig["c3_rsi_zone"],
            "c4_near_high":    sig["c4_near_high"],
            "c5_vol_surge":    sig["c5_vol_surge"],
            **outcome,
        })
    return results


# ─── Reporting helpers ────────────────────────────────────────────────────────
def _stats(label: str, df: pd.DataFrame):
    if df.empty:
        print(f"\n  {label}: no signals")
        return
    closed   = df[df["outcome"] != "no_data"]
    decided  = closed[closed["outcome"].isin(["target_hit", "stop_loss_hit"])]
    n_wins   = (decided["outcome"] == "target_hit").sum()
    n_losses = (decided["outcome"] == "stop_loss_hit").sum()
    pos_ret  = (closed["return_pct"] > 0).sum()
    wr_ts    = round(100 * n_wins / len(decided), 1)  if len(decided) else 0
    wr_ov    = round(100 * pos_ret / len(closed),  1) if len(closed)  else 0
    avg_ret  = round(closed["return_pct"].mean(), 2)  if len(closed)  else 0
    gw = closed.loc[closed["return_pct"] > 0, "return_pct"].sum()
    gl = -closed.loc[closed["return_pct"] < 0, "return_pct"].sum()
    pf = round(gw / gl, 2) if gl > 0 else float("inf")
    ah = round(decided["days_held"].mean(), 1) if len(decided) else 0
    print(f"\n  ── {label}")
    print(f"     Signals: {len(df)} | Closed: {len(closed)} | "
          f"T/SL decided: {len(decided)} (W:{n_wins} L:{n_losses})")
    print(f"     Win rate (T vs SL):  {wr_ts}%  |  Overall profitable: {wr_ov}%")
    print(f"     Avg return: {avg_ret:+.2f}%   |  Profit Factor: {pf}  |  Avg hold: {ah}d")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("backtest_results", exist_ok=True)
    backtest_start = date.today() - timedelta(days=int(BACKTEST_MONTHS * 31))

    print(f"\n{'='*65}")
    print(f"  SHORT-SWING STRATEGY — WALK-FORWARD BACKTEST")
    print(f"  Universe : {len(SYMBOLS)} NSE stocks")
    print(f"  Window   : {backtest_start} → {date.today()}  ({BACKTEST_MONTHS} months)")
    print(f"  Target   : +{TARGET_PCT}%  |  Stop: -{STOP_PCT}%  |  Max hold: {MAX_HOLDING_DAYS}d")
    print(f"  Entry    : Next day open (no look-ahead bias)")
    print(f"  Min cond : {MIN_CONDITIONS}/5 conditions required")
    print(f"{'='*65}\n")

    all_results: list = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_symbol, sym, backtest_start): sym for sym in SYMBOLS}
        for fut in as_completed(futures):
            try:
                all_results.extend(fut.result())
            except Exception as e:
                logger.warning("Worker error: %s", e)
            done += 1
            if done % 10 == 0 or done == len(SYMBOLS):
                print(f"  [{done}/{len(SYMBOLS)}] stocks processed...", flush=True)

    if not all_results:
        print("\nNo signals generated. Try extending BACKTEST_MONTHS or lowering MIN_CONDITIONS.")
        return

    df = pd.DataFrame(all_results)
    out_path = "backtest_results/short_swing_backtest.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} signals → {out_path}")

    # ─── Report ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ACCURACY REPORT")
    print(f"{'='*65}")

    _stats(f"ALL signals ({MIN_CONDITIONS}+ conditions)", df)

    strict = df[df["conditions_met"] == 5]
    _stats("STRICT — all 5 conditions", strict)

    top5_strict = (strict.groupby("signal_date")
                   .apply(lambda g: g.sort_values("adx", ascending=False).head(5))
                   .reset_index(drop=True))
    _stats("STRICT top-5/day (sorted by ADX)", top5_strict)

    top3_strict = (strict.groupby("signal_date")
                   .apply(lambda g: g.sort_values("adx", ascending=False).head(3))
                   .reset_index(drop=True))
    _stats("STRICT top-3/day (sorted by ADX)", top3_strict)

    # Breakdown by conditions met
    print(f"\n  Breakdown by conditions_met:")
    for cond in sorted(df["conditions_met"].unique(), reverse=True):
        sub = df[df["conditions_met"] == cond]
        dec = sub[sub["outcome"].isin(["target_hit", "stop_loss_hit"])]
        w   = (dec["outcome"] == "target_hit").sum()
        nc  = len(sub[sub["outcome"] != "no_data"])
        pos = (sub[sub["outcome"] != "no_data"]["return_pct"] > 0).sum()
        wr  = round(100 * w / len(dec), 1) if len(dec) else 0
        wro = round(100 * pos / nc, 1) if nc else 0
        print(f"    {cond}/5 → {len(sub):3d} signals | T/SL win: {wr}% | Overall profitable: {wro}%")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
