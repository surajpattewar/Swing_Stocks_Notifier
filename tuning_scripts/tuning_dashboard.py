import os
import sys
import pandas as pd
import streamlit as st
from datetime import datetime

# Ensure the root of the project is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config
from tuning_scripts.tuning_db_tracker import get_tuning_signals

# Configure Page
st.set_page_config(
    page_title="Swing Screener Tuning & Simulation",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Strategy Parameter Tuning & Backtest Simulator")
st.markdown("### NSE Swing Screener Optimization Platform")
st.caption("Experiment with technical score thresholds, holding periods, stop-loss ATR coefficients, and risk-reward ratios dynamically.")

# Caching price database loader
@st.cache_data(ttl=60, show_spinner="Loading price database from DuckDB...")
def load_all_stock_prices():
    import duckdb
    db_path = config.DUCKDB_PATH
    if os.path.exists(db_path):
        try:
            with duckdb.connect(db_path, read_only=True) as con:
                df = con.execute("SELECT symbol, date, open, high, low, close FROM stock_prices").fetchdf()
                df["date"] = pd.to_datetime(df["date"])
                if df["date"].dt.tz is not None:
                    df["date"] = df["date"].dt.tz_convert(None)
                return df
        except Exception as e:
            st.error(f"Error loading prices from DuckDB: {e}")
    return pd.DataFrame()

# Caching tuning signals loader
@st.cache_data(ttl=5, show_spinner="Loading tuning signals from Google Sheet...")
def load_tuning_signals_df():
    try:
        df = get_tuning_signals()
        if not df.empty:
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["stop_loss"] = pd.to_numeric(df["stop_loss"], errors="coerce")
            df["target"] = pd.to_numeric(df["target"], errors="coerce")
            df["beta"] = pd.to_numeric(df["beta"], errors="coerce").fillna(0.0)
            df["adx"] = pd.to_numeric(df["adx"], errors="coerce").fillna(0.0)
            df["atr"] = pd.to_numeric(df["atr"], errors="coerce").fillna(0.0)
            df["date"] = pd.to_datetime(df["date"])
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_convert(None)
        return df
    except Exception as e:
        st.error(f"Error loading tuning signals: {e}")
        return pd.DataFrame()

# Dynamic Simulator Engine
def run_ui_simulation(signals_df, prices_df, min_score, max_holding_days, atr_mult, risk_reward):
    sim_results = []
    prices_by_sym = {sym: group.sort_values("date") for sym, group in prices_df.groupby("symbol")}
    
    filtered_signals = signals_df[signals_df["score"] >= min_score]
    if filtered_signals.empty:
        return pd.DataFrame()
        
    for _, signal in filtered_signals.iterrows():
        symbol = signal["symbol"]
        signal_date = signal["date"]
        entry_price = float(signal["close"])
        atr_val = float(signal["atr"])
        
        if atr_val <= 0:
            atr_val = entry_price * 0.02
            
        new_sl = round(entry_price - atr_mult * atr_val, 2)
        risk = max(entry_price - new_sl, 0.01)
        new_target = round(entry_price + risk_reward * risk, 2)
        
        if symbol not in prices_by_sym:
            continue
            
        sym_prices = prices_by_sym[symbol]
        fwd_prices = sym_prices[sym_prices["date"] >= signal_date]
        if fwd_prices.empty:
            continue
            
        fwd_prices = fwd_prices.head(max_holding_days + 1)
        if len(fwd_prices) < 2:
            continue
            
        outcome = "TIMEOUT"
        exit_price = fwd_prices.iloc[-1]["close"]
        exit_date = fwd_prices.iloc[-1]["date"]
        days_held = len(fwd_prices) - 1
        
        for idx in range(1, len(fwd_prices)):
            row = fwd_prices.iloc[idx]
            low_p = float(row["low"])
            high_p = float(row["high"])
            
            # Check outcome: stop hit takes precedence over target hit on same session
            if low_p <= new_sl:
                outcome = "STOP_LOSS"
                exit_price = new_sl
                exit_date = row["date"]
                days_held = idx
                break
            elif high_p >= new_target:
                outcome = "TARGET"
                exit_price = new_target
                exit_date = row["date"]
                days_held = idx
                break
                
        return_pct = (exit_price - entry_price) / entry_price * 100
        
        sim_results.append({
            "symbol": symbol,
            "signal_date": signal_date.date(),
            "score": int(signal["score"]),
            "setup_type": signal["setup_type"],
            "entry_price": entry_price,
            "stop_loss": new_sl,
            "target": new_target,
            "outcome": outcome,
            "exit_price": exit_price,
            "exit_date": exit_date.date(),
            "days_held": days_held,
            "return_pct": round(return_pct, 2)
        })
        
    return pd.DataFrame(sim_results)

# UI Widgets - Sidebar Configuration
st.sidebar.header("⚙️ Simulator Configuration")
st.sidebar.subheader("Adjust Parameters to Tune Strategy")

min_score_input = st.sidebar.slider("Minimum Technical Score Threshold", min_value=3, max_value=15, value=10, step=1, help="Filter out candidates below this score.")
max_holding_days_input = st.sidebar.slider("Max Holding Period (Trading Days)", min_value=1, max_value=30, value=15, step=1, help="Max days to hold a position before exit timeout.")
atr_mult_input = st.sidebar.slider("Stop-Loss ATR Multiplier Coefficient", min_value=1.0, max_value=4.0, value=2.0, step=0.1, help="SL = Entry - Multiplier * ATR.")
risk_reward_input = st.sidebar.slider("Risk-Reward Ratio Coefficient", min_value=1.0, max_value=5.0, value=1.5, step=0.1, help="Target = Entry + Ratio * Risk.")

# Load resources
signals_df = load_tuning_signals_df()
prices_df = load_all_stock_prices()

if signals_df.empty:
    st.info("The 'tuning_sheet' worksheet in Google Sheets is currently empty. Run uv run tuning_scripts/run_tuning_backfill.py to populate it with Nifty 200 signals first!")
elif prices_df.empty:
    st.error("DuckDB price database not found. Please ensure data_ingestion.py has been run to populate data/duckdb/screener_data.duckdb.")
else:
    # Run Simulator
    sim_df = run_ui_simulation(
        signals_df=signals_df,
        prices_df=prices_df,
        min_score=min_score_input,
        max_holding_days=max_holding_days_input,
        atr_mult=atr_mult_input,
        risk_reward=risk_reward_input
    )
    
    if sim_df.empty:
        st.warning("No signals qualified under the selected Minimum Technical Score.")
    else:
        # Stats Calculations
        total_sim = len(sim_df)
        wins_sim = sim_df[sim_df["outcome"] == "TARGET"]
        losses_sim = sim_df[sim_df["outcome"] == "STOP_LOSS"]
        timeouts_sim = sim_df[sim_df["outcome"] == "TIMEOUT"]
        
        pos_wins = sim_df[sim_df["return_pct"] > 0]
        win_rate_pct = (len(pos_wins) / total_sim) * 100
        
        target_hits = len(wins_sim)
        stop_hits = len(losses_sim)
        if (target_hits + stop_hits) > 0:
            target_win_rate = (target_hits / (target_hits + stop_hits)) * 100
            target_wr_str = f"{target_win_rate:.1f}%"
        else:
            target_wr_str = "N/A"
            
        avg_ret_sim = sim_df["return_pct"].mean()
        
        gross_prof = sim_df[sim_df["return_pct"] > 0]["return_pct"].sum()
        gross_loss = abs(sim_df[sim_df["return_pct"] <= 0]["return_pct"].sum())
        p_factor = gross_prof / gross_loss if gross_loss > 0 else (gross_prof if gross_prof > 0 else 1.0)
        
        st.markdown("#### 📊 Tuned Strategy Performance Summary")
        cols_metrics = st.columns(4)
        cols_metrics[0].metric("Overall Win Rate (+ve exit)", f"{win_rate_pct:.1f}%", f"Target vs SL WR: {target_wr_str}")
        cols_metrics[1].metric("Avg Return / Trade", f"{avg_ret_sim:+.2f}%")
        cols_metrics[2].metric("Profit Factor", f"{p_factor:.2f}")
        cols_metrics[3].metric("Tuned Trade Count", f"{total_sim}", f"Target: {target_hits} | SL: {stop_hits} | Timeout: {len(timeouts_sim)}")
        
        st.divider()
        
        # Cumulative returns timeline
        df_cum = sim_df.groupby("exit_date")["return_pct"].sum().reset_index()
        df_cum = df_cum.sort_values(by="exit_date", ascending=True)
        df_cum["cumulative_return"] = df_cum["return_pct"].cumsum()
        
        st.subheader("📈 Tuned Cumulative Return Timeline")
        st.caption("Visualizes growth curve of return percentage over time under the selected parameters.")
        st.line_chart(
            df_cum.set_index("exit_date")[["cumulative_return"]],
            use_container_width=True
        )
        
        st.divider()
        
        # Display simulated trade log
        st.subheader("📜 Tuned Trade Log")
        st.dataframe(
            sim_df[["symbol", "signal_date", "score", "entry_price", "stop_loss", "target", "outcome", "exit_price", "exit_date", "days_held", "return_pct"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "entry_price": st.column_config.NumberColumn("Entry (₹)", format="₹%.2f"),
                "stop_loss": st.column_config.NumberColumn("SL (₹)", format="₹%.2f"),
                "target": st.column_config.NumberColumn("Target (₹)", format="₹%.2f"),
                "exit_price": st.column_config.NumberColumn("Exit (₹)", format="₹%.2f"),
                "return_pct": st.column_config.NumberColumn("Return (%)", format="%+.2f%%"),
            }
        )
