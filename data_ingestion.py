"""
Populates the local `stock_prices` table in DuckDB so the screener and the
backtester can both read price history off disk instead of hitting yfinance
every time.

Usage:
    from data_ingestion import ingest_history, ingest_deltas
    ingest_history(["RELIANCE.NS", "TCS.NS", ...])   # first-time bulk load (~12mo)
    ingest_deltas(["RELIANCE.NS", "TCS.NS", ...])     # incremental top-up (only new days)

Run directly to backfill the current stock universe:
    uv run data_ingestion.py
"""
import os
import logging

import pandas as pd
import yfinance as yf
import duckdb

from config import config

logger = logging.getLogger(__name__)


def _ensure_db_dir(db_path: str):
    d = os.path.dirname(db_path)
    if d:
        os.makedirs(d, exist_ok=True)


def _create_table_if_missing(con):
    con.execute("""
    CREATE TABLE IF NOT EXISTS stock_prices (
        date TIMESTAMPTZ NOT NULL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume BIGINT,
        dividends DOUBLE,
        stock_splits DOUBLE,
        symbol VARCHAR NOT NULL,
        PRIMARY KEY (symbol, date)
    )
    """)


def _upsert(con, df: pd.DataFrame):
    """df must have columns: Date/Open/High/Low/Close/Volume/Dividends/Stock Splits/symbol."""
    if df.empty:
        return 0
    con.register("df_view", df)
    con.execute("""
    INSERT INTO stock_prices
    SELECT
        Date as date,
        Open as open,
        High as high,
        Low as low,
        Close as close,
        Volume as volume,
        Dividends as dividends,
        CAST("Stock Splits" AS DOUBLE) AS stock_splits,
        symbol as symbol
    FROM df_view
    ON CONFLICT(symbol, date)
    DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        dividends = EXCLUDED.dividends,
        stock_splits = EXCLUDED.stock_splits;
    """)
    con.unregister("df_view")
    return len(df)


def ingest_history(symbols, db_path: str = None, period: str = "12mo"):
    """
    First-time / full bulk load: fetches `period` of daily history for every
    symbol and upserts it into stock_prices. Safe to re-run (ON CONFLICT
    updates existing rows rather than duplicating them).
    """
    db_path = db_path or config.DUCKDB_PATH
    _ensure_db_dir(db_path)

    all_dfs = []
    for symbol in symbols:
        try:
            logger.info("Fetching %s history for %s", period, symbol)
            df_history = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
            df_today = yf.Ticker(symbol).history(period="1d", interval="1d", auto_adjust=True)
            df_temp = pd.concat([df_history[:-1], df_today])
            if df_temp.empty:
                logger.warning("No data returned for %s, skipping", symbol)
                continue
            df_temp["symbol"] = symbol
            all_dfs.append(df_temp)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", symbol, e)

    if not all_dfs:
        logger.warning("Nothing fetched, stock_prices left unchanged.")
        return 0

    df = pd.concat(all_dfs).reset_index()

    con = duckdb.connect(db_path)
    try:
        _create_table_if_missing(con)
        n = _upsert(con, df)
        logger.info("Upserted %d rows into stock_prices (%s)", n, db_path)
        return n
    finally:
        con.close()


def ingest_deltas(symbols, db_path: str = None):
    """
    Incremental top-up: for each symbol, finds the latest date already
    stored in stock_prices and fetches only the days since then (falling
    back to a 12mo full fetch if the symbol has no rows yet). Much cheaper
    than re-running ingest_history() every day.
    """
    db_path = db_path or config.DUCKDB_PATH
    _ensure_db_dir(db_path)

    con = duckdb.connect(db_path)
    try:
        _create_table_if_missing(con)

        all_dfs = []
        for symbol in symbols:
            try:
                last_date_row = con.execute(
                    "SELECT max(date) FROM stock_prices WHERE symbol = ?", [symbol]
                ).fetchone()
                last_date = last_date_row[0] if last_date_row else None

                if last_date is None:
                    logger.info("%s has no rows yet, doing a full 12mo fetch", symbol)
                    df_history = yf.Ticker(symbol).history(period="12mo", interval="1d", auto_adjust=True)
                else:
                    # yfinance's `start` is inclusive, so re-fetching last_date's own
                    # bar is harmless (ON CONFLICT upserts it) and guards against a
                    # partial/stale last row from an interrupted prior run.
                    start_str = pd.Timestamp(last_date).strftime("%Y-%m-%d")
                    logger.info("%s last stored date is %s, fetching deltas since then", symbol, start_str)
                    df_history = yf.Ticker(symbol).history(start=start_str, interval="1d", auto_adjust=True)

                if df_history.empty:
                    logger.info("No new rows for %s", symbol)
                    continue

                df_history["symbol"] = symbol
                all_dfs.append(df_history)
            except Exception as e:
                logger.warning("Failed to fetch deltas for %s: %s", symbol, e)

        if not all_dfs:
            logger.info("No deltas to ingest.")
            return 0

        df = pd.concat(all_dfs).reset_index()
        n = _upsert(con, df)
        logger.info("Upserted %d delta rows into stock_prices (%s)", n, db_path)
        return n
    finally:
        con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from stock_universe import get_stock_universe
    syms = get_stock_universe(max_stocks=config.BACKTEST_MAX_STOCKS)
    ingest_history(syms)