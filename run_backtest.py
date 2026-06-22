"""
CLI entry point for backtesting the screener against historical data.

Usage examples:
    uv run run_backtest.py
    uv run run_backtest.py --months 4 --min-score 3 --max-holding-days 15
    uv run run_backtest.py --symbols RELIANCE.NS,TCS.NS,INFY.NS
    uv run run_backtest.py --max-stocks 150 --no-save

What this does:
  1. Builds a stock universe from local DuckDB data, or uses a
     comma-separated list you pass in.
  2. For every trading day across the last N months, re-scores each stock
     using ONLY data available up to that day (walk-forward, no look-ahead).
  3. For every signal that would have fired (score >= min-score), simulates
     forward what actually happened to price afterwards (target hit? stop
     hit? timed out?).
  4. Prints an accuracy report (win rate, avg return, profit factor, win
     rate by score) and persists every signal + outcome to DuckDB.

See backtest.py's module docstring for important caveats before trusting
the numbers (survivorship bias, no fundamental-rule in backtest, fills are
idealized at exact target/stop price, etc).
"""
import argparse
import logging
import sys

from config import config
from database import ScreenerDB
from backtest import (
    run_backtest, compute_accuracy_metrics, print_accuracy_report,
    get_latest_price_date, get_local_symbols,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Backtest the NSE swing screener.")
    p.add_argument("--months", type=int, default=config.BACKTEST_MONTHS,
                    help="How many months back to backtest (default: %(default)s)")
    p.add_argument("--min-score", type=int, default=config.BACKTEST_MIN_SCORE,
                    help="Minimum score to count as a signal (default: %(default)s)")
    p.add_argument("--max-holding-days", type=int, default=config.BACKTEST_MAX_HOLDING_DAYS,
                    help="Max trading days to hold before calling it a timeout (default: %(default)s)")
    p.add_argument("--max-stocks", type=int, default=config.BACKTEST_MAX_STOCKS,
                    help="Cap the universe size to keep runtime reasonable (default: %(default)s)")
    p.add_argument("--symbols", type=str, default=None,
                    help="Comma-separated symbols to test instead of all locally stored symbols, "
                         "e.g. RELIANCE.NS,TCS.NS,INFY.NS")
    p.add_argument("--db-path", type=str, default=config.DUCKDB_PATH,
                    help="DuckDB file containing stock_prices (default: %(default)s)")
    p.add_argument("--max-workers", type=int, default=config.BACKTEST_MAX_WORKERS,
                    help="Parallel symbol workers (default: %(default)s)")
    p.add_argument("--no-save", action="store_true",
                    help="Don't persist results to DuckDB, just print the report")
    p.add_argument("--export-csv", type=str, default=None,
                    help="Optional path to also dump all signals+outcomes as CSV")
    return p.parse_args()


def _progress(done, total, last_symbol):
    if done % 10 == 0 or done == total:
        logger.info("Backtest progress: %d/%d symbols processed (last: %s)",
                    done, total, last_symbol)


def main():
    args = parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = get_local_symbols(args.db_path, max_stocks=args.max_stocks)

    if not symbols:
        logger.error("No symbols found in %s", args.db_path)
        return 1
    as_of = get_latest_price_date(args.db_path, symbols)

    logger.info("Backtesting %d symbols over the last %d months "
                "through local data date %s (min_score=%d, max_holding_days=%d)...",
                len(symbols), args.months, as_of, args.min_score, args.max_holding_days)

    results_df = run_backtest(
        symbols=symbols,
        backtest_months=args.months,
        min_score=args.min_score,
        max_holding_days=args.max_holding_days,
        max_workers=args.max_workers,
        progress_callback=_progress,
        db_path=args.db_path,
    )

    metrics = compute_accuracy_metrics(results_df)
    print_accuracy_report(metrics)

    if args.export_csv:
        results_df.to_csv(args.export_csv, index=False)
        logger.info("Exported %d rows to %s", len(results_df), args.export_csv)

    if not args.no_save and config.SAVE_TO_DUCKDB:
        with ScreenerDB(args.db_path) as db:
            run_id = db.start_run(
                run_type="backtest",
                params={
                    "months": args.months, "min_score": args.min_score,
                    "max_holding_days": args.max_holding_days,
                    "n_symbols": len(symbols), "as_of": str(as_of),
                },
            )
            saved = 0
            for _, row in results_df.iterrows():
                # Build a lightweight candidate-row insert (reuses the candidates
                # table so the signal itself is queryable, then links the outcome).
                cid = db.con.execute("SELECT nextval('candidate_id_seq')").fetchone()[0]
                db.con.execute(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [cid, run_id, row["signal_date"], row["symbol"], row["score"],
                     row["reasons"], row["entry_price"], row["rsi"], row["stop_loss"], row["target"]],
                )
                db.save_backtest_outcome(
                    run_id=run_id, candidate_id=cid, symbol=row["symbol"],
                    signal_date=row["signal_date"], score=row["score"],
                    entry_price=row["entry_price"], stop_loss=row["stop_loss"],
                    target=row["target"],
                    outcome={"outcome": row["outcome"], "exit_price": row["exit_price"],
                             "exit_date": row["exit_date"], "days_held": row["days_held"],
                             "return_pct": row["return_pct"]},
                )
                saved += 1
            logger.info("Persisted %d signals to %s (run_id=%d)", saved, args.db_path, run_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
