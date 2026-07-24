
# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
import os
import sys
import pandas as pd
import duckdb
import ta

# Add workspace to sys.path
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from config import config
from btst_screener import evaluate_btst

def main():
    db_path = config.DUCKDB_PATH
    con = duckdb.connect(db_path, read_only=True)
    
    target_symbols = [
        "BAJAJ-AUTO.NS",
        "TVSMOTOR.NS",
        "NESTLEIND.NS",
        "SHRIRAMFIN.NS"
    ]
    
    latest_date = pd.Timestamp("2026-07-21").date()
    
    # Load Nifty index
    index_raw = con.execute("SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date, close AS Close FROM stock_prices WHERE symbol = 'NSEI' ORDER BY date").fetchdf()
    index_df = pd.DataFrame()
    if not index_raw.empty:
        index_raw["Date"] = pd.to_datetime(index_raw["Date"])
        index_df = index_raw.set_index("Date")
        
    print(f"============================================================")
    print(f"DEBUGGING BTST SCREENER FOR TOP GAINERS ON {latest_date}")
    print(f"============================================================\n")
    
    for symbol in target_symbols:
        # Load price history up to latest_date
        df_raw = con.execute(
            """
            SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                   open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume,
                   delivery_pct
            FROM stock_prices
            WHERE symbol = ? AND CAST(timezone('Asia/Kolkata', date) AS DATE) <= ?
            ORDER BY date
            """,
            [symbol, latest_date]
        ).fetchdf()
        
        if df_raw.empty:
            # Try without .NS suffix
            df_raw = con.execute(
                """
                SELECT CAST(timezone('Asia/Kolkata', date) AS DATE) AS Date,
                       open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume,
                       delivery_pct
                FROM stock_prices
                WHERE symbol = ? AND CAST(timezone('Asia/Kolkata', date) AS DATE) <= ?
                ORDER BY date
                """,
                [symbol.replace(".NS", ""), latest_date]
            ).fetchdf()
            
        if df_raw.empty:
            print(f"❌ {symbol}: Symbol not found in database.")
            continue
            
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df = df_raw.set_index("Date")
        
        if len(df) < 50:
            print(f"❌ {symbol}: Not enough historical bars ({len(df)}) in database.")
            continue
            
        # Run step-by-step trace of evaluate_btst logic to see exactly where it fails or passes
        last = df.iloc[-1]
        
        # Calculate indicators manually
        df["sma20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["sma50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["vol_avg20"] = df["Volume"].shift(1).rolling(20).mean()
        df["rsi14"] = ta.momentum.rsi(df["Close"], window=14)
        df["turnover_avg20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
        df["deliv_avg20"] = df["delivery_pct"].shift(1).rolling(20).mean()
        df["atr"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
        
        last = df.iloc[-1]
        
        # Let's run trace
        rejection_reasons = []
        
        # 1. Trend check
        uptrend = last["Close"] > last["sma20"] and last["Close"] > last["sma50"]
        if not uptrend:
            rejection_reasons.append(f"Not in uptrend (Close ₹{last['Close']:.2f} <= SMA20 ₹{last['sma20']:.2f} or SMA50 ₹{last['sma50']:.2f})")
            
        # 2. Closes near high check
        # Load weights for custom parameter
        from btst_screener import CUSTOM_WEIGHTS
        cfg = CUSTOM_WEIGHTS.get(symbol, {})
        nh_pct = cfg.get("near_high_pct", 0.005)
        vr_limit = cfg.get("vol_ratio_limit", 1.5)
        min_ret = cfg.get("min_return", 1.5)
        rsi_min = cfg.get("rsi_min", 60)
        rsi_max = cfg.get("rsi_max", 78)
        idx_filter = cfg.get("index_filter", "sma20")
        
        near_high = last["Close"] >= (1.0 - nh_pct) * last["High"]
        off_high_pct = (last["High"] - last["Close"]) / last["High"] * 100
        if not near_high:
            rejection_reasons.append(f"Close not near High ({off_high_pct:.2f}% off high, limit {nh_pct*100:.2f}%)")
            
        # 3. Volume spike check
        vol_ratio = last["Volume"] / last["vol_avg20"] if last["vol_avg20"] > 0 else 0
        if vol_ratio < vr_limit:
            rejection_reasons.append(f"No volume spike ({vol_ratio:.2f}x avg, limit {vr_limit}x)")
            
        # 4. Return check
        today_ret = (last["Close"] - last["Open"]) / last["Open"] * 100
        if last["Close"] <= last["Open"] or today_ret < min_ret:
            rejection_reasons.append(f"Not a strong green candle (today's return {today_ret:+.2f}%, limit {min_ret}%)")
            
        # 5. RSI check
        if not (rsi_min <= last["rsi14"] <= rsi_max):
            rejection_reasons.append(f"RSI out of momentum range ({last['rsi14']:.1f}, target {rsi_min}-{rsi_max})")
            
        # 6. Delivery check (ensure absolute delivery volume is higher than average)
        if "delivery_pct" in last and "deliv_avg20" in last and "vol_avg20" in last and not pd.isna(last["delivery_pct"]) and not pd.isna(last["deliv_avg20"]):
            margin = config.BTST_DELIVERY_MARGIN
            today_deliv_vol = last["delivery_pct"] * last["Volume"]
            avg_deliv_vol = last["deliv_avg20"] * last["vol_avg20"]
            delivery_ok = today_deliv_vol >= margin * avg_deliv_vol
            if not delivery_ok:
                rejection_reasons.append(f"Delivery volume low (Today's Absolute Deliv Vol {today_deliv_vol:.1f} <= {margin}x average {avg_deliv_vol:.1f})")
                
        # 7. Circuit safety check
        close_near_high = (last["High"] - last["Close"]) / last["Close"] <= 0.0005
        narrow_range = (last["High"] - last["Low"]) / last["Close"] < 0.005
        if close_near_high and narrow_range:
            rejection_reasons.append(f"Excluded due to circuit lock / narrow day")
            
        # 8. Liquidity safety check
        if "turnover_avg20" in last and last["turnover_avg20"] < 10000000:
            rejection_reasons.append(f"Low turnover ({last['turnover_avg20']/10000000:.2f} Crore < 1 Crore)")
            
        # Print results
        if not rejection_reasons:
            print(f"✅ {symbol}: Triggered BTST candidate signal!")
            print(f"   CMP: ₹{last['Close']:.2f} | Today's Return: {today_ret:+.2f}% | RSI: {last['rsi14']:.1f}")
        else:
            print(f"❌ {symbol}: Filtered out. Reasons:")
            for reason in rejection_reasons:
                print(f"   - {reason}")
        print("-" * 60)
        
    con.close()

if __name__ == "__main__":
    main()
