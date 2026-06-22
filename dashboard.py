"""Streamlit dashboard for persisted DuckDB backtest results."""
import os

import duckdb
import pandas as pd
import streamlit as st

from config import config
from dashboard_metrics import compute_dashboard_metrics, format_metric, parse_params


@st.cache_data(ttl=30, show_spinner=False)
def load_backtest_runs(db_path: str) -> pd.DataFrame:
    with duckdb.connect(db_path, read_only=True) as con:
        return con.execute("""
            SELECT r.run_id, r.run_timestamp, r.params, count(o.id) AS signals
            FROM runs r
            LEFT JOIN backtest_outcomes o ON o.run_id = r.run_id
            WHERE r.run_type = 'backtest'
            GROUP BY r.run_id, r.run_timestamp, r.params
            ORDER BY r.run_timestamp DESC
        """).fetchdf()


@st.cache_data(ttl=30, show_spinner=False)
def load_outcomes(db_path: str, run_id: int) -> pd.DataFrame:
    with duckdb.connect(db_path, read_only=True) as con:
        return con.execute("""
            SELECT o.symbol, o.signal_date, o.score, c.reasons,
                   o.entry_price, o.stop_loss, o.target, o.outcome,
                   o.exit_price, o.exit_date, o.days_held, o.return_pct
            FROM backtest_outcomes o
            LEFT JOIN candidates c ON c.id = o.candidate_id
            WHERE o.run_id = ?
            ORDER BY o.signal_date, o.symbol
        """, [run_id]).fetchdf()


def render_dashboard():
    st.set_page_config(page_title="Swing Backtest Dashboard", layout="wide")
    st.title("Swing Backtest Dashboard")
    st.caption("Results loaded exclusively from the local DuckDB database.")

    default_db = os.getenv("DUCKDB_PATH", config.DUCKDB_PATH)
    with st.sidebar:
        st.header("Data source")
        db_path = st.text_input("DuckDB path", value=default_db)
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if not os.path.isfile(db_path):
        st.error(f"DuckDB file not found: {db_path}")
        st.stop()

    try:
        runs = load_backtest_runs(db_path)
    except (duckdb.Error, OSError) as exc:
        st.error(f"Could not read DuckDB: {exc}")
        st.stop()

    if runs.empty:
        st.info("No saved backtest runs found. Run `uv run run_backtest.py` first.")
        st.stop()

    run_labels = {
        int(row.run_id): (
            f"Run {int(row.run_id)} | {row.run_timestamp:%Y-%m-%d %H:%M} | "
            f"{int(row.signals)} signals"
        )
        for row in runs.itertuples()
    }
    with st.sidebar:
        st.header("Backtest run")
        run_id = st.selectbox(
            "Run", options=list(run_labels), format_func=run_labels.get,
        )

    selected_run = runs.loc[runs["run_id"] == run_id].iloc[0]
    params = parse_params(selected_run["params"])
    try:
        outcomes = load_outcomes(db_path, int(run_id))
    except (duckdb.Error, OSError) as exc:
        st.error(f"Could not load outcomes: {exc}")
        st.stop()

    if outcomes.empty:
        st.info("This run has no saved signals.")
        st.stop()

    outcomes["signal_date"] = pd.to_datetime(outcomes["signal_date"])
    outcomes["exit_date"] = pd.to_datetime(outcomes["exit_date"])
    min_date = outcomes["signal_date"].min().date()
    max_date = outcomes["signal_date"].max().date()

    with st.sidebar:
        st.header("Filters")
        symbols = st.multiselect(
            "Symbols", sorted(outcomes["symbol"].dropna().unique()),
            placeholder="All symbols",
        )
        outcome_options = sorted(outcomes["outcome"].dropna().unique())
        selected_outcomes = st.multiselect(
            "Outcomes", outcome_options, default=outcome_options,
        )
        score_range = st.slider(
            "Score", int(outcomes["score"].min()), int(outcomes["score"].max()),
            (int(outcomes["score"].min()), int(outcomes["score"].max())),
        )
        date_range = st.date_input(
            "Signal dates", value=(min_date, max_date),
            min_value=min_date, max_value=max_date,
        )

    filtered = outcomes[
        outcomes["outcome"].isin(selected_outcomes)
        & outcomes["score"].between(*score_range)
    ].copy()
    if symbols:
        filtered = filtered[filtered["symbol"].isin(symbols)]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        filtered = filtered[
            filtered["signal_date"].dt.date.between(date_range[0], date_range[1])
        ]

    st.caption(
        " · ".join([
            f"Run {run_id}",
            f"As of {params.get('as_of', 'unknown')}",
            f"{params.get('months', '?')} months",
            f"Min score {params.get('min_score', '?')}",
            f"Max hold {params.get('max_holding_days', '?')} days",
        ])
    )

    metrics = compute_dashboard_metrics(filtered)
    metric_cols = st.columns(6)
    metric_cols[0].metric("Signals", f"{metrics['signals']:,}")
    metric_cols[1].metric("Target hits", f"{metrics['target_hits']:,}")
    metric_cols[2].metric("Stop hits", f"{metrics['stop_hits']:,}")
    metric_cols[3].metric(
        "Target vs stop win rate", format_metric(metrics["target_win_rate"], "%")
    )
    metric_cols[4].metric("Average return", format_metric(metrics["avg_return"], "%", 2))
    metric_cols[5].metric("Profit factor", format_metric(metrics["profit_factor"], "", 2))

    if filtered.empty:
        st.warning("No signals match the selected filters.")
        st.stop()

    left, right = st.columns(2)
    with left:
        st.subheader("Cumulative signal returns")
        cumulative = (
            filtered.dropna(subset=["return_pct"])
            .groupby("signal_date", as_index=False)["return_pct"].sum()
            .sort_values("signal_date")
        )
        cumulative["cumulative_return_pct"] = cumulative["return_pct"].cumsum()
        st.line_chart(cumulative, x="signal_date", y="cumulative_return_pct")
    with right:
        st.subheader("Outcome distribution")
        distribution = filtered["outcome"].value_counts().rename_axis("outcome").reset_index(name="signals")
        st.bar_chart(distribution, x="outcome", y="signals")

    left, right = st.columns(2)
    with left:
        st.subheader("Performance by score")
        by_score = (
            filtered.groupby("score", as_index=False)
            .agg(signals=("symbol", "size"), avg_return_pct=("return_pct", "mean"))
        )
        st.bar_chart(by_score, x="score", y="avg_return_pct")
        st.dataframe(by_score, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Performance by symbol")
        by_symbol = (
            filtered.groupby("symbol", as_index=False)
            .agg(signals=("symbol", "size"), avg_return_pct=("return_pct", "mean"))
            .sort_values(["avg_return_pct", "signals"], ascending=False)
        )
        st.dataframe(
            by_symbol, hide_index=True, use_container_width=True,
            column_config={"avg_return_pct": st.column_config.NumberColumn(format="%.2f%%")},
        )

    st.subheader("Signal details")
    display = filtered.sort_values(["signal_date", "symbol"], ascending=[False, True])
    st.dataframe(
        display, hide_index=True, use_container_width=True,
        column_config={"return_pct": st.column_config.NumberColumn(format="%.2f%%")},
    )
    st.download_button(
        "Download filtered CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name=f"backtest_run_{run_id}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    render_dashboard()
