import os
import pandas as pd
import streamlit as st
from config import config
from db_tracker import get_daily_results, get_open_positions, get_position_progress

# Configure Page
st.set_page_config(
    page_title="Swing Trading Live Tracker",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Swing Trading Live Tracker Dashboard")
st.caption("Real-time position tracking, daily screener archive, and sequential day-wise progress from your Google Sheets database.")

@st.cache_data(ttl=5, show_spinner=False)
def load_db_summary():
    summary = {}
    try:
        df_daily = get_daily_results()
        df_open = get_open_positions()
        
        summary["total_candidates"] = len(df_daily)
        
        if not df_open.empty:
            df_open["entry_price"] = pd.to_numeric(df_open["entry_price"], errors="coerce")
            df_open["close_price"] = pd.to_numeric(df_open["close_price"], errors="coerce")
            
            summary["active_trades"] = len(df_open[df_open["status"] == "OPEN"])
            df_closed = df_open[df_open["status"] == "CLOSED"]
            summary["closed_trades"] = len(df_closed)
            
            if len(df_closed) > 0:
                wins = (df_closed["close_price"] > df_closed["entry_price"]).sum()
                summary["win_rate"] = (wins / len(df_closed)) * 100
                returns = ((df_closed["close_price"] - df_closed["entry_price"]) / df_closed["entry_price"]) * 100
                summary["avg_return"] = returns.mean()
            else:
                summary["win_rate"] = 0.0
                summary["avg_return"] = 0.0
        else:
            summary["active_trades"] = 0
            summary["closed_trades"] = 0
            summary["win_rate"] = 0.0
            summary["avg_return"] = 0.0
            
    except Exception as e:
        st.error(f"Error loading summary: {e}")
        summary = {"total_candidates": 0, "active_trades": 0, "closed_trades": 0, "win_rate": 0.0, "avg_return": 0.0}
    return summary

@st.cache_data(ttl=5, show_spinner=False)
def load_active_positions():
    try:
        df_open = get_open_positions()
        df_prog = get_position_progress()
        
        if df_open.empty:
            return pd.DataFrame()
            
        df_active = df_open[df_open["status"] == "OPEN"].copy()
        if df_active.empty:
            return pd.DataFrame()
            
        df_active["entry_price"] = pd.to_numeric(df_active["entry_price"], errors="coerce")
        df_active["current_price"] = pd.to_numeric(df_active["current_price"], errors="coerce")
        df_active["current_sl"] = pd.to_numeric(df_active["current_sl"], errors="coerce")
        df_active["current_target"] = pd.to_numeric(df_active["current_target"], errors="coerce")
        
        if not df_prog.empty:
            df = pd.merge(df_active, df_prog, on="symbol", how="left")
        else:
            df = df_active
        return df
    except Exception as e:
        st.error(f"Error loading active positions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5, show_spinner=False)
def load_closed_positions():
    try:
        df_open = get_open_positions()
        if df_open.empty:
            return pd.DataFrame()
            
        df_closed = df_open[df_open["status"] == "CLOSED"].copy()
        if df_closed.empty:
            return pd.DataFrame()
            
        df_closed["entry_price"] = pd.to_numeric(df_closed["entry_price"], errors="coerce")
        df_closed["close_price"] = pd.to_numeric(df_closed["close_price"], errors="coerce")
        df_closed["return_pct"] = round(((df_closed["close_price"] - df_closed["entry_price"]) / df_closed["entry_price"]) * 100, 2)
        
        # Sort by close_date descending
        df_closed = df_closed.sort_values(by="close_date", ascending=False)
        return df_closed
    except Exception as e:
        st.error(f"Error loading closed positions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5, show_spinner=False)
def load_daily_results():
    try:
        df = get_daily_results()
        if df.empty:
            return pd.DataFrame()
            
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["stop_loss"] = pd.to_numeric(df["stop_loss"], errors="coerce")
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        
        # Sort by date and score descending
        df = df.sort_values(by=["date", "score"], ascending=[False, False])
        return df
    except Exception as e:
        st.error(f"Error loading daily results: {e}")
        return pd.DataFrame()

# Refresh action
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Connection Info
st.sidebar.markdown(f"**Database Connection:** `GOOGLE SHEETS`")
st.sidebar.caption(f"Connected to sheet: `{config.GOOGLE_SHEET_NAME}`.")

# 1. Summary Metrics
summary = load_db_summary()
cols = st.columns(5)
cols[0].metric("Total Screened Candidates", f"{summary['total_candidates']}")
cols[1].metric("Active Open Trades", f"{summary['active_trades']}")
cols[2].metric("Closed Trades", f"{summary['closed_trades']}")
cols[3].metric("Strategy Win Rate", f"{summary['win_rate']:.1f}%")
cols[4].metric("Average Return / Trade", f"{summary['avg_return']:+.2f}%")

# 2. Main Sections Tabs
tab1, tab2, tab3 = st.tabs([
    "📂 Active Positions & 15-Day Progress",
    "📅 Daily Screener Archive",
    "📜 Closed Trade History"
])

with tab1:
    st.subheader("Active Positions & Sequential Progress Timeline")
    df_active = load_active_positions()
    if df_active.empty:
        st.info("No active open positions found in the database. Run the daily screener to open positions.")
    else:
        st.caption("Daywise columns note % change from entry price, target/stop hits, and timeouts.")
        
        # Display progress table with nice styling
        display_cols = [
            "symbol", "entry_date", "entry_price", "current_price", "current_return", "current_sl", "current_target", "setup_type",
            "day1", "day2", "day3", "day4", "day5", "day6", "day7", "day8", "day9", "day10", "day11", "day12", "day13", "day14", "day15"
        ]
        
        # Format columns dynamically
        st.dataframe(
            df_active[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "entry_price": st.column_config.NumberColumn("Entry (₹)", format="₹%.2f"),
                "current_price": st.column_config.NumberColumn("Current (₹)", format="₹%.2f"),
                "current_return": "Current Return",
                "current_sl": st.column_config.NumberColumn("SL (₹)", format="₹%.2f"),
                "current_target": st.column_config.NumberColumn("Target (₹)", format="₹%.2f"),
                "entry_date": "Opened Date",
                "setup_type": "Setup Category",
            }
        )

with tab2:
    st.subheader("Daily Screener Findings History")
    df_daily = load_daily_results()
    if df_daily.empty:
        st.info("No daily candidate results found in the database.")
    else:
        # Date selector
        available_dates = df_daily["date"].dropna().unique()
        selected_date = st.selectbox("Select Date to View Findings", options=available_dates)
        
        filtered_daily = df_daily[df_daily["date"] == selected_date]
        st.dataframe(
            filtered_daily[["symbol", "setup_type", "score", "close", "stop_loss", "target", "reasons"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "close": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
                "stop_loss": st.column_config.NumberColumn("SL (₹)", format="₹%.2f"),
                "target": st.column_config.NumberColumn("Target (₹)", format="₹%.2f"),
            }
        )

with tab3:
    st.subheader("Historical Closed Swing Trades")
    df_closed = load_closed_positions()
    if df_closed.empty:
        st.info("No closed trade records found in the database.")
    else:
        st.dataframe(
            df_closed[["symbol", "entry_date", "entry_price", "close_date", "close_price", "setup_type", "return_pct"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "entry_price": st.column_config.NumberColumn("Entry (₹)", format="₹%.2f"),
                "close_price": st.column_config.NumberColumn("Exit (₹)", format="₹%.2f"),
                "return_pct": st.column_config.NumberColumn("Return (%)", format="%+.2f%%"),
            }
        )
