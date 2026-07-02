import os
import sys
import re
import glob
import time
import itertools
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the workspace root to sys.path to resolve local imports
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from screener import evaluate, Candidate
from stock_universe import get_stock_universe, FALLBACK_SYMBOLS
import backtest_yfinance

# Configure Page
st.set_page_config(
    page_title="Swing Screener Backtest Console 📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Premium aesthetics */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-weight: 700 !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1f6feb 0%, #094cb4 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #388bfd 0%, #1f6feb 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.4);
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #58a6ff;
        margin-bottom: 0.2rem;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-size: 16px;
        font-weight: 600;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to scan for pre-generated runs
def get_historical_runs():
    os.makedirs("backtest_results", exist_ok=True)
    signal_files = glob.glob("backtest_results/backtest_signals_*.csv")
    runs = []
    for f in signal_files:
        match = re.search(r"backtest_signals_(\d{8}_\d{6})\.csv", f)
        if match:
            ts = match.group(1)
            summary_f = f"backtest_results/backtest_summary_{ts}.csv"
            if os.path.exists(summary_f):
                try:
                    dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                    runs.append({
                        "timestamp": ts,
                        "datetime": dt,
                        "signals_file": f,
                        "summary_file": summary_f,
                        "label": f"{dt.strftime('%Y-%m-%d %H:%M:%S')} ({ts})"
                    })
                except Exception:
                    pass
    return sorted(runs, key=lambda x: x["datetime"], reverse=True)

# Compute KPIs from signals
def get_overall_kpis(signals_df):
    if signals_df.empty:
        return {"signals": 0, "symbols": 0, "win_rate": 0.0, "avg_return": 0.0, "profit_factor": 1.0}
    
    total_signals = len(signals_df)
    symbols_count = signals_df["symbol"].nunique()
    
    closed = signals_df[signals_df["outcome"] != "no_data"]
    decided = closed[closed["outcome"].isin(["target_hit", "stop_loss_hit"])]
    n_decided = len(decided)
    n_wins = (decided["outcome"] == "target_hit").sum()
    win_rate = (100.0 * n_wins / n_decided) if n_decided > 0 else 0.0
    
    closed_for_pct = closed.dropna(subset=["return_pct"])
    avg_return = closed_for_pct["return_pct"].mean() if not closed_for_pct.empty else 0.0
    
    gross_gain = closed_for_pct.loc[closed_for_pct["return_pct"] > 0, "return_pct"].sum()
    gross_loss = -closed_for_pct.loc[closed_for_pct["return_pct"] < 0, "return_pct"].sum()
    
    profit_factor = gross_gain / gross_loss if gross_loss > 0 else (99.9 if gross_gain > 0 else 1.0)
    
    return {
        "signals": total_signals,
        "symbols": symbols_count,
        "win_rate": round(win_rate, 2),
        "avg_return": round(avg_return, 2),
        "profit_factor": round(profit_factor, 2)
    }

# Render KPI cards
def render_kpis(kpis):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{kpis["signals"]:,}</div><div class="metric-lbl">Total Signals</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{kpis["symbols"]}</div><div class="metric-lbl">Unique Symbols</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{kpis["win_rate"]}%</div><div class="metric-lbl">Win Rate (Tgt vs Stop)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{kpis["avg_return"]:+.2f}%</div><div class="metric-lbl">Avg Return per Trade</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{kpis["profit_factor"]}</div><div class="metric-lbl">Profit Factor</div></div>', unsafe_allow_html=True)

# Render results section for loaded dataframe and summary
def render_results_view(signals_df, summary_df, run_label=""):
    if signals_df.empty:
        st.warning("No signals to display.")
        return

    st.subheader(f"Results for: {run_label if run_label else 'Current Run'}")
    
    # Calculate KPIs
    kpis = get_overall_kpis(signals_df)
    render_kpis(kpis)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create Tab layout for reports
    rep_tab1, rep_tab2, rep_tab3, rep_tab4 = st.tabs([
        "🎯 Individual Indicator Stats", 
        "🧩 Stacking & Combinations", 
        "📈 Return Charts & Analysis",
        "📋 Detailed Trades Log"
    ])
    
    with rep_tab1:
        st.subheader("Performance of Individual Screener Pointers")
        st.markdown("Metrics for each pointer calculated stand-alone on this signal set:")
        
        # Filter for rows that are "Single: <pointer_name>"
        if not summary_df.empty:
            singles = summary_df[summary_df["combination"].str.startswith("Single:")].copy()
            if not singles.empty:
                singles["combination"] = singles["combination"].str.replace("Single: ", "")
                st.dataframe(
                    singles.rename(columns={"combination": "Screener Pointer"}),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "win_rate_target_vs_stop_pct": st.column_config.NumberColumn("Win Rate (Tgt vs Stop)", format="%.1f%%"),
                        "win_rate_overall_pct": st.column_config.NumberColumn("Overall Win Rate", format="%.1f%%"),
                        "avg_return_pct": st.column_config.NumberColumn("Avg Return", format="%.2f%%"),
                        "profit_factor": st.column_config.NumberColumn("Profit Factor", format="%.2f")
                    }
                )
            else:
                st.info("No single pointer entries found in summary data.")
        else:
            st.info("Summary dataframe is empty.")

    with rep_tab2:
        st.subheader("Top Indicator Combinations & Stacked Scores")
        st.markdown("Shows how combining multiple pointers affects metrics. Minimum signal filters apply:")
        
        if not summary_df.empty:
            combos = summary_df[~summary_df["combination"].str.startswith("Single:")].copy()
            
            # Interactive filters for combos
            min_signals = st.slider("Minimum signals count filter", min_value=1, max_value=100, value=5)
            filtered_combos = combos[combos["total_signals"] >= min_signals]
            
            st.dataframe(
                filtered_combos,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "combination": st.column_config.TextColumn("Pointer Stacking Combination"),
                    "total_signals": st.column_config.NumberColumn("Total Signals"),
                    "win_rate_target_vs_stop_pct": st.column_config.NumberColumn("Win Rate (Tgt/Stop)", format="%.1f%%"),
                    "win_rate_overall_pct": st.column_config.NumberColumn("Overall Win Rate", format="%.1f%%"),
                    "avg_return_pct": st.column_config.NumberColumn("Avg Return", format="%.2f%%"),
                    "profit_factor": st.column_config.NumberColumn("Profit Factor", format="%.2f")
                }
            )
        else:
            st.info("Summary dataframe is empty.")

    with rep_tab3:
        st.subheader("Performance Charts & Visualizations")
        
        # Prepare charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Cumulative Returns Over Time")
            # Group return by signal date to see the equity curve
            signals_df["parsed_date"] = pd.to_datetime(signals_df["signal_date"])
            cum_returns = signals_df.dropna(subset=["return_pct"]).groupby("parsed_date")["return_pct"].mean().sort_index().reset_index()
            cum_returns["cumulative_return_pct"] = cum_returns["return_pct"].cumsum()
            
            st.line_chart(cum_returns.set_index("parsed_date")["cumulative_return_pct"])
            
        with chart_col2:
            st.markdown("#### Outcome Breakdown")
            outcomes_dist = signals_df["outcome"].value_counts().reset_index()
            outcomes_dist.columns = ["Outcome", "Count"]
            st.bar_chart(outcomes_dist.set_index("Outcome")["Count"])
            
        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            st.markdown("#### Average Return by Signal Score")
            score_perf = signals_df.groupby("score")["return_pct"].mean().reset_index()
            st.bar_chart(score_perf.set_index("score")["return_pct"])
            
        with chart_col4:
            st.markdown("#### Top 20 Most Profitable Symbols")
            sym_perf = signals_df.groupby("symbol").agg(
                signals_count=("return_pct", "size"),
                avg_return=("return_pct", "mean")
            ).sort_values("avg_return", ascending=False).head(20).reset_index()
            st.dataframe(
                sym_perf,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "avg_return": st.column_config.NumberColumn("Avg Return", format="%.2f%%")
                }
            )

    with rep_tab4:
        st.subheader("Filter and Search Individual Signals")
        
        # Signals filtering options
        filt_col1, filt_col2, filt_col3 = st.columns(3)
        with filt_col1:
            search_symbol = st.multiselect("Filter by Symbols", sorted(signals_df["symbol"].unique()), placeholder="All symbols")
        with filt_col2:
            selected_outcomes = st.multiselect("Filter by Outcomes", sorted(signals_df["outcome"].unique()), default=sorted(signals_df["outcome"].unique()))
        with filt_col3:
            score_range = st.slider("Filter by Score Range", int(signals_df["score"].min()), int(signals_df["score"].max()), (int(signals_df["score"].min()), int(signals_df["score"].max())))
            
        filtered_df = signals_df[
            (signals_df["outcome"].isin(selected_outcomes)) & 
            (signals_df["score"].between(score_range[0], score_range[1]))
        ]
        
        if search_symbol:
            filtered_df = filtered_df[filtered_df["symbol"].isin(search_symbol)]
            
        st.write(f"Showing {len(filtered_df):,} out of {len(signals_df):,} total signals")
        st.dataframe(
            filtered_df.drop(columns=["parsed_date"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
            column_config={
                "entry_price": st.column_config.NumberColumn(format="%.2f"),
                "stop_loss": st.column_config.NumberColumn(format="%.2f"),
                "target": st.column_config.NumberColumn(format="%.2f"),
                "exit_price": st.column_config.NumberColumn(format="%.2f"),
                "return_pct": st.column_config.NumberColumn("Return %", format="%.2f%%")
            }
        )
        
        # Download filtered data
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Signals as CSV",
            data=csv_data,
            file_name="backtest_filtered_results.csv",
            mime="text/csv"
        )

# App Title & Layout
st.title("Swing Screener Backtest Console 📊")
st.markdown("Trigger multi-pointer walk-forward backtests on custom universes, customize screener criteria, and explore historic performance profiles in detail.")
st.markdown("---")

tab_run, tab_history = st.tabs(["🚀 Run New Backtest", "📂 View Historical Runs"])

with tab_run:
    st.header("Configure and Execute Simulation")
    
    # Left inputs and right info card layout
    left_input, right_desc = st.columns([2, 1])
    
    with left_input:
        universe_choice = st.selectbox(
            "Target Stock Universe",
            options=["NSE 100 Index Stocks", "Custom List", "Single Symbol", "Fallback Liquid Tickers (Top 50)"],
            index=0,
            help="Select which stocks to download historical prices and perform simulation for."
        )
        
        symbols_to_run = []
        if universe_choice == "NSE 100 Index Stocks":
            max_stocks = st.slider("Limit maximum stocks to fetch (saves API rate limit/time)", min_value=5, max_value=100, value=100)
            st.info(f"Will retrieve and run backtest on up to the top {max_stocks} stocks of the Nifty 100.")
        elif universe_choice == "Custom List":
            custom_input = st.text_area("Input symbols (comma-separated)", value="TCS.NS, INFY.NS, RELIANCE.NS, AXISBANK.NS, COALINDIA.NS")
            st.caption("E.g. RELIANCE.NS, TCS.NS, INFY.NS (use .NS extension for NSE stocks)")
        elif universe_choice == "Single Symbol":
            custom_input = st.text_input("Input symbol", value="RELIANCE.NS")
        elif universe_choice == "Fallback Liquid Tickers (Top 50)":
            st.info("Will use the pre-defined fallback list of 50 highly liquid stocks.")
            
        st.markdown("#### Backtest Period & Criteria")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            years_to_test = st.slider("Backtest Period (Years)", min_value=0.5, max_value=10.0, value=5.0, step=0.5)
        with col_p2:
            min_score = st.slider("Minimum Score", min_value=1, max_value=6, value=1, help="Min score to capture a trade. Select 1 to test performance of all individual indicator triggers.")
        with col_p3:
            max_hold = st.slider("Max holding period (trading days)", min_value=5, max_value=30, value=15)
            
        max_workers = st.slider("Parallel workers (download speed)", min_value=1, max_value=20, value=10)
        
        # Cache control checkboxes
        cache_path = "data/stock_history_cache.pkl"
        cache_exists = os.path.exists(cache_path)
        cache_info = ""
        if cache_exists:
            try:
                import pickle
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                    cached_symbols_count = len(cache_data.get("stock_data", {}))
                    mtime = os.path.getmtime(cache_path)
                    last_mod = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    cache_info = f"💾 Cache: {cached_symbols_count} symbols, last updated {last_mod}"
            except Exception:
                cache_info = "⚠️ Cache file exists but failed to read."
        else:
            cache_info = "ℹ️ No local cache found (will fetch from Yahoo Finance)."

        st.markdown(f"**Cache Status**: {cache_info}")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            use_cache = st.checkbox("Use local cache (if available)", value=cache_exists, help="Load historical data locally rather than fetching from Yahoo Finance.")
        with col_c2:
            save_cache = st.checkbox("Save/Update cache after run", value=True, help="Update cache with any new downloaded symbols.")

    with right_desc:
        st.markdown("""
        ### Walk-Forward Rules
        1. **Daily Incremental Scan**: The engine advances day-by-day. At each day, it extracts the trailing 130 trading days (~200 calendar days) of price history.
        2. **Signals Generation**: It invokes the improved `screener.py` `evaluate` logic to verify structural conditions (Uptrends, Pullbacks, OLS Breakouts, Support touch, RSI zones, etc.).
        3. **Outcome Tracking**: If structural score $\ge$ Min Score, the engine logs a signal and simulates price resolution up to `Max holding period`.
        4. **Conservative Resolution**: If a day's High hits target and Low hits stop loss simultaneously, it is conservatively resolved as a **Stop Loss hit** (negative return).
        """)

    # Action button
    run_btn = st.button("🚀 Start Backtest Simulation", use_container_width=True)
    
    if run_btn:
        # Prepare list of symbols based on choice
        symbols = []
        if universe_choice == "NSE 100 Index Stocks":
            with st.spinner("Fetching NSE 100 Stocks list..."):
                symbols = get_stock_universe(max_stocks=max_stocks, no_of_stocks=100)
        elif universe_choice == "Custom List":
            symbols = [s.strip().upper() for s in custom_input.split(",") if s.strip()]
        elif universe_choice == "Single Symbol":
            symbols = [custom_input.strip().upper()]
        elif universe_choice == "Fallback Liquid Tickers (Top 50)":
            symbols = [f"{s}.NS" for s in FALLBACK_SYMBOLS]
            
        # Clean index symbols
        symbols = [s for s in symbols if not s.startswith("^")]
        
        if not symbols:
            st.error("No valid symbols supplied for simulation.")
        else:
            # Let's perform simulation and update UI
            status_container = st.status("Initializing backtest...", expanded=True)
            
            with status_container:
                # 1. Download price data
                progress_bar = st.progress(0.0)
                
                # Fetch data
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=int(years_to_test * 365))
                warmup_start_date = start_date - timedelta(days=200)
                
                start_str = warmup_start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                
                stock_data = {}
                betas = {}
                symbols_to_download = list(symbols)
                
                if use_cache:
                    st.write("Loading historical data from local cache...")
                    cached_data, cached_betas, missing = backtest_yfinance.load_cached_data(symbols)
                    stock_data.update(cached_data)
                    betas.update(cached_betas)
                    symbols_to_download = missing
                    st.write(f"Loaded {len(cached_data)} symbols from local cache. {len(missing)} symbols require downloading.")
                
                if symbols_to_download:
                    st.write(f"Downloading historical stock price files for {len(symbols_to_download)} symbols...")
                    try:
                        if len(symbols_to_download) == 1:
                            df_all = yf.download(symbols_to_download[0], start=start_str, end=end_str, interval="1d", auto_adjust=True, progress=False)
                            if not df_all.empty:
                                if df_all.index.tz is not None:
                                    df_all.index = df_all.index.tz_localize(None)
                                stock_data[symbols_to_download[0]] = df_all
                        else:
                            df_all = yf.download(symbols_to_download, start=start_str, end=end_str, interval="1d", auto_adjust=True, group_by='ticker', progress=False)
                            for sym in symbols_to_download:
                                if sym in df_all:
                                    df_sym = df_all[sym].dropna(how='all')
                                    if not df_sym.empty and len(df_sym) >= 130:
                                        if df_sym.index.tz is not None:
                                            df_sym.index = df_sym.index.tz_localize(None)
                                        stock_data[sym] = df_sym
                    except Exception as ex:
                        st.error(f"Download failed: {ex}")
                
                progress_bar.progress(0.4)
                
                if not stock_data:
                    st.error("No data available for simulation. Please check your ticker formatting and network connection.")
                else:
                    # Download missing betas
                    missing_betas_symbols = [sym for sym in stock_data.keys() if sym not in betas]
                    if missing_betas_symbols:
                        st.write(f"Fetched historical data. Now downloading beta metrics for {len(missing_betas_symbols)} symbols...")
                        for idx, sym in enumerate(missing_betas_symbols, 1):
                            beta = 0.0
                            try:
                                ticker = yf.Ticker(sym)
                                beta = float(ticker.info.get("beta") or 0.0)
                            except Exception:
                                pass
                            betas[sym] = beta
                            time.sleep(0.05)
                    
                    progress_bar.progress(0.6)
                    st.write("Executing walk-forward logic (evaluating daily screeners)...")
                    
                    all_signals = []
                    
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {}
                        for sym, df in stock_data.items():
                            beta = betas[sym]
                            futures[executor.submit(
                                backtest_yfinance.process_symbol, 
                                sym, df, beta, start_date, end_date, min_score, max_hold
                            )] = sym
                            
                        processed_cnt = 0
                        for fut in as_completed(futures):
                            sym = futures[fut]
                            processed_cnt += 1
                            try:
                                res = fut.result()
                                all_signals.extend(res)
                            except Exception as e:
                                st.warning(f"Error processing symbol {sym}: {e}")
                                
                            # Update progress
                            pct = 0.6 + (0.3 * (processed_cnt / len(stock_data)))
                            progress_bar.progress(min(pct, 0.9))
                            
                    progress_bar.progress(0.9)
                    
                    signals_df = pd.DataFrame(all_signals)
                    if signals_df.empty:
                        status_container.update(label="Simulation Finished: No Signals Generated", state="complete")
                        st.info("The screener produced 0 trade signals with the current criteria.")
                    else:
                        st.write("Analyzing pointer combinations and compiling summary metrics...")
                        signals_df = signals_df.sort_values(by=["signal_date", "symbol"]).reset_index(drop=True)
                        
                        # Generate permutation file
                        analysis_df = backtest_yfinance.analyze_permutations(signals_df)
                        
                        # Save result files to backtest_results/
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        os.makedirs("backtest_results", exist_ok=True)
                        signals_file = os.path.join("backtest_results", f"backtest_signals_{timestamp}.csv")
                        summary_file = os.path.join("backtest_results", f"backtest_summary_{timestamp}.csv")
                        
                        signals_df.to_csv(signals_file, index=False)
                        analysis_df.to_csv(summary_file, index=False)
                        
                        if save_cache:
                            st.write("Updating local cache file...")
                            backtest_yfinance.save_cached_data(stock_data, betas)
                        
                        progress_bar.progress(1.0)
                        status_container.update(label="Simulation Complete! 🎉", state="complete", expanded=False)
                        
                        # Store in session state for instant view
                        st.session_state["last_signals_df"] = signals_df
                        st.session_state["last_summary_df"] = analysis_df
                        st.session_state["last_run_label"] = f"Live Run: {len(stock_data)} symbols, {len(signals_df):,} signals"

    # Display live run results if they exist in session state
    if "last_signals_df" in st.session_state:
        render_results_view(
            st.session_state["last_signals_df"], 
            st.session_state["last_summary_df"], 
            st.session_state["last_run_label"]
        )


with tab_history:
    st.header("Pregenerated Backtest Runs Dashboard")
    st.markdown("Select a previously computed run to view details, accuracy breakdowns, and stacking performance:")
    
    runs = get_historical_runs()
    
    if not runs:
        st.info("No historical backtest runs found in `backtest_results/` directory.")
    else:
        # Selection option
        selected_run = st.selectbox(
            "Select historical backtest run",
            options=runs,
            format_func=lambda x: x["label"]
        )
        
        if selected_run:
            with st.spinner("Loading run data from CSV files..."):
                try:
                    signals_df = pd.read_csv(selected_run["signals_file"])
                    summary_df = pd.read_csv(selected_run["summary_file"])
                    
                    render_results_view(
                        signals_df, 
                        summary_df, 
                        f"Historical Run {selected_run['timestamp']}"
                    )
                except Exception as e:
                    st.error(f"Failed to load historical run files: {e}")
