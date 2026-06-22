"""
Walk-forward backtester for the swing screener.

What "backtesting" means here, precisely:
  For each trading day `t` in the backtest window (default: last 4 months),
  re-run the EXACT SAME scoring logic from screener.py using only price data
  that was available up to and including day `t` (no peeking at the future).
  If a stock scores >= min_score on day t, that's a "signal" with an entry
  price, stop-loss, and target (computed the same way screener.py does it
  live). We then look FORWARD from day t (data the screener didn't see) and
  check which came first:
    - High >= target    -> target_hit    (a "win")
    - Low  <= stop_loss  -> stop_loss_hit (a "loss")
    - neither within `max_holding_days` -> timeout (closed at last close)
  Accuracy = how often target_hit happens before stop_loss_hit, i.e. how
  often the screener's own suggested trade would have worked out.

Important caveats (read before trusting the numbers):
  1. Universe survivorship: unless an explicit symbol list is supplied, we
     backtest the symbols currently present in the local stock_prices table.
     This may still omit delisted/renamed stocks and flatter results.
  2. Book-value rule (rule 6 in screener.py) is intentionally SKIPPED during
     backtesting (see screener.evaluate(skip_fundamental=True)) because
     yfinance only exposes the CURRENT book value, not the historical one —
     using it for a signal 3 months ago would be look-ahead bias. So
     backtested scores run 0-6, not 0-7. Live and backtested scores aren't
     perfectly apples-to-apples for that reason.
  3. Slippage/costs: target/stop fills are simulated as if you got filled
     exactly at the target/stop-loss price, intraday, the moment High/Low
     crosses it. Real fills will be a bit worse (gaps, slippage, brokerage).
  4. This evaluates the screener's *signal quality*, not a full portfolio
     backtest (no position sizing, capital constraints, or overlapping-
     trade capital limits are modeled).
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import yfinance as yf

import duckdb
import pandas as pd

from config import config
from screener import evaluate, Candidate

logger = logging.getLogger(__name__)

# How many calendar days of prior history the indicators need (SMA100 is the
# longest lookback in screener.py) before the first evaluable day. Padded.
INDICATOR_WARMUP_CALENDAR_DAYS = 160


def get_local_symbols(db_path: str = None, max_stocks: int = None) -> list[str]:
    """Return the symbol universe already stored in the local price table."""
    db_path = db_path or config.DUCKDB_PATH
    limit_sql = " LIMIT ?" if max_stocks is not None else ""
    params = [max_stocks] if max_stocks is not None else []
    with duckdb.connect(db_path, read_only=True) as con:
        rows = con.execute(
            "SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol" + limit_sql,
            params,
        ).fetchall()
    return [row[0] for row in rows]


def get_latest_price_date(db_path: str = None, symbols: list[str] = None) -> date:
    """Return the newest locally stored bar date for the requested universe."""
    db_path = db_path or config.DUCKDB_PATH
    where_sql = ""
    params = []
    if symbols:
        where_sql = f" WHERE symbol IN ({','.join('?' for _ in symbols)})"
        params = symbols
    with duckdb.connect(db_path, read_only=True) as con:
        row = con.execute(
            "SELECT max(CAST(timezone('Asia/Kolkata', date) AS DATE)) "
            "FROM stock_prices" + where_sql,
            params,
        ).fetchone()
    if not row or row[0] is None:
        raise ValueError(f"No local price data found in {db_path}")
    return pd.Timestamp(row[0]).date()


def fetch_long_history(symbol: str, backtest_months: int, max_holding_days: int,
                       db_path: str = None, as_of_date: date = None) -> pd.DataFrame:
    """
    Load one symbol's indicator warmup and backtest window exclusively from
    the local DuckDB stock_prices table. This function never uses the network.
    """
    db_path = db_path or config.DUCKDB_PATH
    end = as_of_date or get_latest_price_date(db_path, [symbol])
    calendar_days_needed = (
        INDICATOR_WARMUP_CALENDAR_DAYS
        + int(backtest_months * 31)
        + int(max_holding_days * 1.6) + 10
    )
    start = end - timedelta(days=calendar_days_needed)
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute(
            """
            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                   open AS Open, high AS High, low AS Low,
                   close AS Close, volume AS Volume, dividends AS Dividends,
                   stock_splits AS "Stock Splits"
            FROM stock_prices
            WHERE symbol = ?
              AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
            ORDER BY date
            """,
            [symbol, start, end],
        ).fetchdf()
    if df.empty:
        raise ValueError(f"No local history found for {symbol} in {db_path}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df

# def fetch_long_history(
#     symbol: str,
#     backtest_months: int,
#     max_holding_days: int, db_path: str = None, as_of_date: date = None
# ) -> pd.DataFrame:
#
#     calendar_days_needed = (
#         INDICATOR_WARMUP_CALENDAR_DAYS
#         + int(backtest_months * 31)
#         + int(max_holding_days * 1.6)
#         + 10
#     )
#
#     years = max(2, int(calendar_days_needed / 365) + 1)
#
#     df = yf.Ticker(symbol).history(
#         period=f"{years}y",
#         interval="1d",
#         auto_adjust=True,
#     )
#
#     if df.empty:
#         raise ValueError(f"No data for {symbol}")
#
#     return df

def simulate_outcome(df: pd.DataFrame, signal_date, entry_price: float,
                      stop_loss: float, target: float, max_holding_days: int) -> dict:
    """
    Look forward from signal_date (exclusive) and see whether target or
    stop_loss is hit first, using daily High/Low. If both trigger on the
    same day we conservatively assume stop_loss is hit first (daily bars
    can't tell us the actual intraday order).
    """
    future = df.loc[df.index > signal_date].iloc[:max_holding_days]

    if future.empty:
        return {"outcome": "no_data", "exit_price": None, "exit_date": None,
                "days_held": 0, "return_pct": None}

    for i, (dt, row) in enumerate(future.iterrows(), start=1):
        hit_target = row["High"] >= target
        hit_stop = row["Low"] <= stop_loss
        if hit_stop:  # conservative: stop wins same-day ties
            exit_price = stop_loss
            return {"outcome": "stop_loss_hit", "exit_price": exit_price,
                     "exit_date": dt.date(), "days_held": i,
                     "return_pct": round((exit_price - entry_price) / entry_price * 100, 2)}
        if hit_target:
            exit_price = target
            return {"outcome": "target_hit", "exit_price": exit_price,
                     "exit_date": dt.date(), "days_held": i,
                     "return_pct": round((exit_price - entry_price) / entry_price * 100, 2)}

    last_row = future.iloc[-1]
    exit_price = float(last_row["Close"])
    return {"outcome": "timeout", "exit_price": exit_price,
             "exit_date": future.index[-1].date(), "days_held": len(future),
             "return_pct": round((exit_price - entry_price) / entry_price * 100, 2)}


def walk_forward_signals(symbol: str, df: pd.DataFrame, backtest_start, backtest_end,
                          min_score: int, max_holding_days: int) -> list:
    """
    For every trading day in [backtest_start, backtest_end] that has enough
    prior history, re-score the stock as of that day (no look-ahead) and,
    if it qualifies, simulate the forward outcome.
    Returns a list of dicts (one per qualifying signal).
    """
    results = []
    eval_dates = df.index[(df.index >= pd.Timestamp(backtest_start)) &
                           (df.index <= pd.Timestamp(backtest_end))]

    for signal_date in eval_dates:
        df_slice = df.loc[:signal_date]
        if len(df_slice) < 110:   # not enough warm-up for SMA100 etc.
            continue
        try:
            cand: Candidate = evaluate(symbol, df_slice, skip_fundamental=True)
        except Exception as e:
            logger.debug("Eval failed for %s on %s: %s", symbol, signal_date.date(), e)
            continue

        if cand.score < min_score:
            continue

        outcome = simulate_outcome(
            df, signal_date=signal_date, entry_price=cand.close,
            stop_loss=cand.stop_loss, target=cand.target,
            max_holding_days=max_holding_days,
        )
        results.append({
            "symbol": symbol,
            "signal_date": signal_date.date(),
            "score": cand.score,
            "reasons": "; ".join(cand.reasons),
            "entry_price": cand.close,
            "rsi": cand.rsi,
            "stop_loss": cand.stop_loss,
            "target": cand.target,
            **outcome,
        })
    return results


def run_backtest(symbols: list, backtest_months: int = 4, min_score: int = 3,
                  max_holding_days: int = 15, max_workers: int = 6,
                  progress_callback=None, db_path: str = None) -> pd.DataFrame:
    """
    Runs the walk-forward backtest across all `symbols` and returns a single
    DataFrame of every signal generated + its outcome.
    """
    db_path = db_path or config.DUCKDB_PATH
    end = get_latest_price_date(db_path, symbols)
    start = end - timedelta(days=int(backtest_months * 31))

    all_results = []

    def _process(symbol):
        try:
            df = fetch_long_history(
                symbol, backtest_months, max_holding_days,
                db_path=db_path, as_of_date=end,
            )
            return walk_forward_signals(symbol, df, start, end, min_score, max_holding_days)
        except Exception as e:
            logger.warning("Backtest skipped %s: %s", symbol, e)
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, sym): sym for sym in symbols}
        done = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                res = fut.result()
                all_results.extend(res)
            except Exception as e:
                logger.warning("Backtest error on %s: %s", sym, e)
            done += 1
            if progress_callback:
                progress_callback(done, len(symbols), sym)

    if not all_results:
        return pd.DataFrame(columns=[
            "symbol", "signal_date", "score", "reasons", "entry_price", "rsi",
            "stop_loss", "target", "outcome", "exit_price", "exit_date",
            "days_held", "return_pct",
        ])

    return pd.DataFrame(all_results).sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def compute_accuracy_metrics(results_df: pd.DataFrame) -> dict:
    """
    Summarizes backtest results into headline accuracy numbers.
    """
    if results_df.empty:
        return {"total_signals": 0, "message": "No signals generated in this window."}

    closed = results_df[results_df["outcome"] != "no_data"].copy()
    decided = closed[closed["outcome"].isin(["target_hit", "stop_loss_hit"])]
    timeouts = closed[closed["outcome"] == "timeout"]

    total_signals = len(results_df)
    n_decided = len(decided)
    n_wins = (decided["outcome"] == "target_hit").sum()
    n_losses = (decided["outcome"] == "stop_loss_hit").sum()
    win_rate_decided = round(100 * n_wins / n_decided, 1) if n_decided else None

    # Softer view: did the trade close profitable at all (incl. timeouts)?
    closed_for_pct = closed.dropna(subset=["return_pct"])
    n_profitable_overall = (closed_for_pct["return_pct"] > 0).sum()
    win_rate_overall = (
        round(100 * n_profitable_overall / len(closed_for_pct), 1) if len(closed_for_pct) else None
    )

    avg_return = round(closed_for_pct["return_pct"].mean(), 2) if len(closed_for_pct) else None
    gross_gain = closed_for_pct.loc[closed_for_pct["return_pct"] > 0, "return_pct"].sum()
    gross_loss = -closed_for_pct.loc[closed_for_pct["return_pct"] < 0, "return_pct"].sum()
    profit_factor = round(gross_gain / gross_loss, 2) if gross_loss > 0 else None

    by_score = (
        decided.groupby("score")
        .agg(signals=("outcome", "count"),
             win_rate=("outcome", lambda s: round(100 * (s == "target_hit").mean(), 1)))
        .reset_index()
        .sort_values("score", ascending=False)
    ) if n_decided else pd.DataFrame()

    return {
        "total_signals": total_signals,
        "closed_signals": len(closed),
        "still_open_no_data": total_signals - len(closed),
        "decided_signals": n_decided,
        "target_hits": int(n_wins),
        "stop_loss_hits": int(n_losses),
        "timeouts": len(timeouts),
        "win_rate_target_vs_stop_pct": win_rate_decided,
        "win_rate_overall_pct": win_rate_overall,
        "avg_return_pct_per_signal": avg_return,
        "profit_factor": profit_factor,
        "by_score_breakdown": by_score,
    }


def print_accuracy_report(metrics: dict):
    print("\n" + "=" * 60)
    print("BACKTEST ACCURACY REPORT")
    print("=" * 60)
    if metrics.get("total_signals", 0) == 0:
        print(metrics.get("message", "No signals."))
        return

    print(f"Total signals generated        : {metrics['total_signals']}")
    print(f"  - closed (target/stop/timeout): {metrics['closed_signals']}")
    print(f"  - still open / no fwd data    : {metrics['still_open_no_data']}")
    print(f"  - target hit                  : {metrics['target_hits']}")
    print(f"  - stop-loss hit               : {metrics['stop_loss_hits']}")
    print(f"  - timed out (held to limit)   : {metrics['timeouts']}")
    print("-" * 60)
    print(f"Win rate (target reached before stop)  : {metrics['win_rate_target_vs_stop_pct']}%")
    print(f"Win rate (closed with any +ve return)   : {metrics['win_rate_overall_pct']}%")
    print(f"Average return per signal                : {metrics['avg_return_pct_per_signal']}%")
    print(f"Profit factor (gross win / gross loss)   : {metrics['profit_factor']}")
    print("-" * 60)
    bs = metrics.get("by_score_breakdown")
    if bs is not None and not bs.empty:
        print("Win rate by score:")
        for _, r in bs.iterrows():
            print(f"  score {int(r['score'])}: {int(r['signals'])} signals, "
                  f"{r['win_rate']}% win rate")
    print("=" * 60 + "\n")
