"""
DuckDB persistence layer.

Stores every screener run (live or backtest) as a local .duckdb file so you
can query history later (e.g. "show me every BUY signal on RELIANCE in the
last 6 months", or "what's my screener's win rate by score").

DuckDB is just a file on disk — no server to run. Default path is
`screener_data.duckdb` in the working directory (override via
DUCKDB_PATH in .env / config.py).

NOTE on GitHub Actions: the runner's filesystem is thrown away after each
job, so a .duckdb file written during a run will NOT persist to the next
run unless you either (a) commit it back to the repo as a workflow step,
or (b) point DUCKDB_PATH at a path inside a mounted/persistent volume
(e.g. on a VM or Render disk). For GitHub Actions specifically, add a
"commit and push data/screener_data.duckdb" step after the screener runs,
or upload it as a workflow artifact.
"""
import json
import logging
from datetime import datetime, date

import duckdb

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "screener_data.duckdb"


class ScreenerDB:
    """Thin wrapper around a DuckDB file with the tables this project needs."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _init_schema(self):
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS run_id_seq START 1")
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS candidate_id_seq START 1")
        self.con.execute("CREATE SEQUENCE IF NOT EXISTS outcome_id_seq START 1")

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id          BIGINT PRIMARY KEY,
                run_type        VARCHAR,      -- 'live' or 'backtest'
                run_timestamp   TIMESTAMP,
                params          VARCHAR       -- JSON blob of config used for the run
            )
        """)

        # One row per stock that scored >= min_score on a given signal_date.
        # Used for both live runs and backtest signal generation.
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id            BIGINT PRIMARY KEY,
                run_id        BIGINT,
                signal_date   DATE,
                symbol        VARCHAR,
                score         INTEGER,
                reasons       VARCHAR,
                close         DOUBLE,
                rsi           DOUBLE,
                stop_loss     DOUBLE,
                target        DOUBLE
            )
        """)

        # One row per backtested signal, with what actually happened next.
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS backtest_outcomes (
                id            BIGINT PRIMARY KEY,
                candidate_id  BIGINT,
                run_id        BIGINT,
                symbol        VARCHAR,
                signal_date   DATE,
                score         INTEGER,
                entry_price   DOUBLE,
                stop_loss     DOUBLE,
                target        DOUBLE,
                outcome       VARCHAR,   -- target_hit | stop_loss_hit | timeout | no_data
                exit_price    DOUBLE,
                exit_date     DATE,
                days_held     INTEGER,
                return_pct    DOUBLE
            )
        """)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def start_run(self, run_type: str, params: dict) -> int:
        run_id = self.con.execute("SELECT nextval('run_id_seq')").fetchone()[0]
        self.con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?)",
            [run_id, run_type, datetime.now(), json.dumps(params, default=str)],
        )
        return run_id

    def save_candidates(self, run_id: int, signal_date: date, candidates: list) -> dict:
        """
        Persists a list of screener.Candidate objects for one signal_date.
        Returns {symbol: candidate_id} so callers (e.g. the backtester) can
        link outcomes back to the candidate row.
        """
        ids = {}
        for c in candidates:
            cid = self.con.execute("SELECT nextval('candidate_id_seq')").fetchone()[0]
            self.con.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [cid, run_id, signal_date, c.symbol, c.score, "; ".join(c.reasons),
                 c.close, c.rsi, c.stop_loss, c.target],
            )
            ids[c.symbol] = cid
        return ids

    def save_backtest_outcome(self, run_id: int, candidate_id: int, symbol: str,
                               signal_date: date, score: int, entry_price: float,
                               stop_loss: float, target: float, outcome: dict) -> int:
        oid = self.con.execute("SELECT nextval('outcome_id_seq')").fetchone()[0]
        self.con.execute(
            "INSERT INTO backtest_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [oid, candidate_id, run_id, symbol, signal_date, score, entry_price,
             stop_loss, target, outcome["outcome"], outcome["exit_price"],
             outcome["exit_date"], outcome["days_held"], outcome["return_pct"]],
        )
        return oid

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def get_candidates(self, run_id: int = None):
        q = "SELECT * FROM candidates"
        params = []
        if run_id is not None:
            q += " WHERE run_id = ?"
            params.append(run_id)
        q += " ORDER BY signal_date, score DESC"
        return self.con.execute(q, params).df()

    def get_backtest_outcomes(self, run_id: int = None):
        q = "SELECT * FROM backtest_outcomes"
        params = []
        if run_id is not None:
            q += " WHERE run_id = ?"
            params.append(run_id)
        q += " ORDER BY signal_date, symbol"
        return self.con.execute(q, params).df()

    def get_runs(self):
        return self.con.execute("SELECT * FROM runs ORDER BY run_timestamp DESC").df()

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
