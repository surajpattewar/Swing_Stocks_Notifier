from datetime import date, datetime
import tempfile
import unittest
from pathlib import Path

import duckdb

from backtest import fetch_long_history, get_latest_price_date, get_local_symbols


class LocalBacktestDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.price_db = str(Path(self.temp_dir.name) / "prices.duckdb")
        with duckdb.connect(self.price_db) as con:
            con.execute("""
                CREATE TABLE stock_prices (
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
            con.executemany(
                "INSERT INTO stock_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (datetime.fromisoformat("2026-06-20T00:00:00+05:30"), 100, 110, 90,
                     105, 1000, 0, 0, "AAA.NS"),
                    (datetime.fromisoformat("2026-06-21T00:00:00+05:30"), 105, 112, 101,
                     110, 1200, 0, 0, "AAA.NS"),
                    (datetime.fromisoformat("2026-06-19T00:00:00+05:30"), 50, 55, 48,
                     52, 500, 0, 0, "BBB.NS"),
                ],
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_local_universe_and_latest_date(self):
        self.assertEqual(get_local_symbols(self.price_db), ["AAA.NS", "BBB.NS"])
        self.assertEqual(get_local_symbols(self.price_db, max_stocks=1), ["AAA.NS"])
        self.assertEqual(get_latest_price_date(self.price_db), date(2026, 6, 21))
        self.assertEqual(
            get_latest_price_date(self.price_db, ["BBB.NS"]), date(2026, 6, 19)
        )

    def test_fetch_history_preserves_exchange_calendar_date(self):
        df = fetch_long_history(
            "AAA.NS", backtest_months=1, max_holding_days=5,
            db_path=self.price_db, as_of_date=date(2026, 6, 21),
        )

        self.assertEqual(list(df.index.date), [date(2026, 6, 20), date(2026, 6, 21)])
        self.assertEqual(list(df.columns), [
            "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"
        ])

    def test_fetch_history_rejects_symbol_missing_from_local_db(self):
        with self.assertRaisesRegex(ValueError, "No local history found"):
            fetch_long_history(
                "MISSING.NS", backtest_months=1, max_holding_days=5,
                db_path=self.price_db, as_of_date=date(2026, 6, 21),
            )


if __name__ == "__main__":
    unittest.main()
