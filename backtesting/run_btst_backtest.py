
# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
import os
import json
import duckdb
import pandas as pd
import numpy as np
import ta
from datetime import date, timedelta
from config import config
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_btst_backtest")

MODEL_FILE = "models/btst_stock_weights.json"
CUSTOM_WEIGHTS = {}
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, "r") as f:
            CUSTOM_WEIGHTS = json.load(f)
        print(f"Loaded custom BTST parameters for {len(CUSTOM_WEIGHTS)} stocks.")
    except Exception as e:
        print(f"Failed to load custom weights: {e}")

DB_PATH = "data/duckdb/screener_data.duckdb"

def get_symbols(con):
    rows = con.execute("SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol").fetchall()
    return [r[0] for r in rows if r[0] != 'NSEI']

def load_data(con, symbol, start_date, end_date):
    warmup_days = 100
    start_warm = start_date - timedelta(days=warmup_days)
    df = con.execute(
        """
        SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
               open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume,
               delivery_pct AS DeliveryPct
        FROM stock_prices
        WHERE symbol = ? AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
        ORDER BY date
        """,
        [symbol, start_warm, end_date]
    ).fetchdf()
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")

def run_btst_backtest(symbols_data, start_date, end_date, index_df=None, delivery_margin=None):
    trades = []
    
    # Calculate index SMA50
    idx_df = pd.DataFrame()
    if index_df is not None and not index_df.empty:
        idx_df = index_df.copy()
        idx_df["sma50"] = ta.trend.sma_indicator(idx_df["Close"], window=50)
        
    for symbol, df_all in symbols_data.items():
        if df_all is None or len(df_all) < 60:
            continue
            
        df = df_all.copy()
        df["sma20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["vol_avg20"] = df["Volume"].shift(1).rolling(20).mean()
        df["rsi14"] = ta.momentum.rsi(df["Close"], window=14)
        df["turnover"] = df["Close"] * df["Volume"]
        df["turnover_avg20"] = df["turnover"].shift(1).rolling(20).mean()
        if "DeliveryPct" in df.columns:
            df["deliv_avg20"] = df["DeliveryPct"].shift(1).rolling(20).mean()
        df = df.dropna()
        
        # Get custom parameters or defaults
        params = CUSTOM_WEIGHTS.get(symbol, {
            "near_high_pct": 0.002,
            "vol_ratio_limit": 1.5,
            "min_return": 2.0,
            "rsi_min": 60,
            "rsi_max": 78,
            "index_filter": "sma50"
        })
        nh_pct = params.get("near_high_pct", 0.002)
        vr_limit = params.get("vol_ratio_limit", 1.5)
        min_ret = params.get("min_return", 2.0)
        rsi_min = params.get("rsi_min", 60)
        rsi_max = params.get("rsi_max", 78)
        idx_filter = params.get("index_filter", "sma50")
        
        eval_dates = df.index[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
        
        for idx in range(len(df) - 1):
            dt = df.index[idx]
            if dt not in eval_dates:
                continue
                
            row = df.iloc[idx]
            next_row = df.iloc[idx + 1] # Day D+1
            
            # BTST Screener Criteria using custom/default parameters
            pct_today = (row["Close"] - row["Open"]) / row["Open"] * 100
            near_high = row["Close"] >= (1.0 - nh_pct) * row["High"]
            vol_spike = row["Volume"] >= vr_limit * row["vol_avg20"]
            uptrend = row["Close"] > row["sma20"] and row["Close"] > row["sma50"]
            rsi_momentum = rsi_min <= row["rsi14"] <= rsi_max
            
            # Broader Market Index filter
            index_ok = True
            if idx_filter == "sma50" and not idx_df.empty:
                if dt in idx_df.index and dt in idx_df["sma50"].index:
                    index_ok = idx_df.loc[dt, "Close"] > idx_df.loc[dt, "sma50"]
            
            # Delivery percentage filter (if enabled)
            delivery_ok = True
            if delivery_margin is not None and "DeliveryPct" in df.columns:
                delivery_ok = row["DeliveryPct"] >= delivery_margin * row["deliv_avg20"]

            # Circuit safety check (exceeding high by less than 0.05%, and daily range < 0.5%)
            is_circuit_locked = False
            if row["Close"] > 0:
                close_near_high = (row["High"] - row["Close"]) / row["Close"] <= 0.0005
                narrow_range = (row["High"] - row["Low"]) / row["Close"] < 0.005
                is_circuit_locked = close_near_high and narrow_range
                
            # Liquidity safety check (20-day avg daily turnover >= 1 Crore)
            is_illiquid = row["turnover_avg20"] < 10000000

            if (is_circuit_locked or is_illiquid) and uptrend and near_high and vol_spike and row["Close"] > row["Open"] and pct_today >= min_ret and rsi_momentum and index_ok and delivery_ok:
                if is_circuit_locked:
                    logger.info("BTST Circuit Lock Excluded %s on %s: High-Close range %s, High-Low range %s", symbol, dt.date(), round((row["High"]-row["Close"])/row["Close"]*100, 3), round((row["High"]-row["Low"])/row["Close"]*100, 3))
                if is_illiquid:
                    logger.info("BTST Illiquid Excluded %s on %s: 20d Avg Turnover %s", symbol, dt.date(), round(row["turnover_avg20"], 0))
                continue
            
            if uptrend and near_high and vol_spike and row["Close"] > row["Open"] and pct_today >= min_ret and rsi_momentum and index_ok and delivery_ok:
                entry_price = float(row["Close"])
                next_open = float(next_row["Open"])
                next_high = float(next_row["High"])
                next_low = float(next_row["Low"])
                next_close = float(next_row["Close"])
                
                # Exit Type A: Sell at Next Day's Open (Immediate Morning gap capture)
                ret_open = (next_open - entry_price) / entry_price * 100
                round_trip = config.BACKTEST_TRANSACTION_COST_PCT * 2
                ret_open_net = ret_open - round_trip
                
                # Exit Type B: Limit Target +1.5% or SL -1.5%, else Close
                target_pct = 1.5
                sl_pct = -1.5
                
                target_price = entry_price * (1 + target_pct/100)
                sl_price = entry_price * (1 + sl_pct/100)
                
                if next_open >= target_price:
                    # Gapped up past target, fill at Open
                    ret_limit = (next_open - entry_price) / entry_price * 100
                    outcome = "win"
                elif next_open <= sl_price:
                    # Gapped down past SL, fill at Open
                    ret_limit = (next_open - entry_price) / entry_price * 100
                    outcome = "loss"
                elif next_high >= target_price:
                    ret_limit = target_pct
                    outcome = "win"
                elif next_low <= sl_price:
                    ret_limit = sl_pct
                    outcome = "loss"
                else:
                    # Exit at Close
                    ret_limit = (next_close - entry_price) / entry_price * 100
                    outcome = "win" if ret_limit > 0 else "loss"
                    
                ret_limit_net = ret_limit - round_trip
                
                trades.append({
                    "symbol": symbol,
                    "date": dt.date(),
                    "entry": entry_price,
                    "exit_open": next_open,
                    "exit_close": next_close,
                    "ret_open_raw": round(ret_open, 2),
                    "ret_open": round(ret_open_net, 2),
                    "ret_limit_raw": round(ret_limit, 2),
                    "ret_limit": round(ret_limit_net, 2),
                    "outcome": outcome
                })
                
    return pd.DataFrame(trades)

def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    symbols = get_symbols(con)
    latest_date_row = con.execute("SELECT max(date) FROM stock_prices").fetchone()
    end_date = pd.Timestamp(latest_date_row[0]).date()
    start_date = end_date - timedelta(days=200) # Backtest last 6-7 months (200 days)
    
    print(f"Loading data for {len(symbols)} symbols over last 6-7 months...")
    symbols_data = {}
    for sym in symbols:
        symbols_data[sym] = load_data(con, sym, start_date, end_date)
        
    # Load Nifty index data
    index_df = None
    try:
        index_raw = con.execute(
            """
            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                   close AS Close
            FROM stock_prices
            WHERE symbol = 'NSEI' AND CAST(timezone('Asia/Kolkata', date) AS DATE) BETWEEN ? AND ?
            ORDER BY date
            """,
            [start_date - timedelta(days=100), end_date]
        ).fetchdf()
        if not index_raw.empty:
            index_raw["Date"] = pd.to_datetime(index_raw["Date"])
            index_df = index_raw.set_index("Date")
    except Exception as e:
        print(f"Warning: Could not load Nifty index benchmark from DuckDB: {e}")
        
    con.close()
    
    margins = [None, 1.1, 1.25, 1.5]
    sweep_results = []
    
    for margin in margins:
        df_trades = run_btst_backtest(symbols_data, start_date, end_date, index_df=index_df, delivery_margin=margin)
        if df_trades.empty:
            continue
            
        wins_limit = (df_trades["ret_limit"] > 0).sum()
        win_rate_limit = wins_limit / len(df_trades) * 100
        avg_ret_limit = df_trades["ret_limit"].mean()
        
        gross_win = df_trades.loc[df_trades["ret_limit"] > 0, "ret_limit"].sum()
        gross_loss = -df_trades.loc[df_trades["ret_limit"] < 0, "ret_limit"].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else 99.9
        
        # False Positive Rate: Gap down on next open (ret_open_raw < 0)
        gap_downs = (df_trades["ret_open_raw"] < 0).sum()
        fpr = gap_downs / len(df_trades) * 100
        
        sweep_results.append({
            "margin": f"{margin}x" if margin else "None (Baseline)",
            "trades": len(df_trades),
            "win_rate": f"{win_rate_limit:.2f}%",
            "avg_ret": f"{avg_ret_limit:+.2f}%",
            "profit_factor": f"{pf:.2f}",
            "false_positive_rate": f"{fpr:.2f}%"
        })
        
    print("\n" + "=" * 80)
    print("                    BTST DELIVERY PERCENTAGE MARGIN SWEEP")
    print("=" * 80)
    headers = ["Margin", "Trades", "Win Rate", "Avg Return", "Profit Factor", "False Pos Rate"]
    col_w = [18, 8, 10, 12, 15, 15]
    row_fmt = "  " + "".join(f"{{:<{w}}}" for w in col_w)
    
    print(row_fmt.format(*headers))
    print("  " + "-" * 76)
    for r in sweep_results:
        print(row_fmt.format(
            r["margin"],
            str(r["trades"]),
            r["win_rate"],
            r["avg_ret"],
            r["profit_factor"],
            r["false_positive_rate"]
        ))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
