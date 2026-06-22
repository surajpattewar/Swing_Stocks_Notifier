"""Pure calculations shared by the Streamlit backtest dashboard."""
import json

import pandas as pd


def compute_dashboard_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "signals": 0, "target_hits": 0, "stop_hits": 0,
            "target_win_rate": None, "profitable_rate": None,
            "avg_return": None, "profit_factor": None,
        }

    decided = df[df["outcome"].isin(["target_hit", "stop_loss_hit"])]
    returns = df["return_pct"].dropna()
    target_hits = int((decided["outcome"] == "target_hit").sum())
    stop_hits = int((decided["outcome"] == "stop_loss_hit").sum())
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return {
        "signals": len(df),
        "target_hits": target_hits,
        "stop_hits": stop_hits,
        "target_win_rate": 100 * target_hits / len(decided) if len(decided) else None,
        "profitable_rate": 100 * (returns > 0).mean() if len(returns) else None,
        "avg_return": returns.mean() if len(returns) else None,
        "profit_factor": gains / losses if losses > 0 else None,
    }


def format_metric(value, suffix="", decimals=1) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.{decimals}f}{suffix}"


def parse_params(raw_params: str) -> dict:
    try:
        return json.loads(raw_params or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
