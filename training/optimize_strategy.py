#!/usr/bin/env python3

# Add parent directory to sys.path to resolve imports from root
import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
"""
Optimization and pattern-learning engine for the swing screener.
Uses scikit-learn tree classifiers to extract high-probability rules and indicator importances
from historical backtest logs.
"""
import os
import glob
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optimize_strategy")


def get_latest_signals_file(results_dir: str = "backtest_results") -> str | None:
    """Find the most recent signals CSV file in the results directory."""
    files = glob.glob(os.path.join(results_dir, "backtest_signals_*.csv"))
    if not files:
        return None
    # Sort by filename timestamp
    return max(files, key=os.path.getmtime)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize swing trading indicator rules.")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to signals CSV file (defaults to latest in backtest_results/)")
    parser.add_argument("--min-samples", type=int, default=15,
                        help="Minimum samples required for a rule leaf node (default: 15)")
    parser.add_argument("--max-rules", type=int, default=5,
                        help="Maximum rules to extract (default: 5)")
    return parser.parse_args()


def extract_rules_from_tree(tree, feature_names, node_id=0, current_rule=None) -> list[dict]:
    """Recursively traverse a decision tree to extract rules leading to high win-rate nodes."""
    if current_rule is None:
        current_rule = []

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    
    # Check if leaf node
    if left_child == right_child:
        # leaf
        n_samples = int(tree.n_node_samples[node_id])
        values = tree.value[node_id][0]
        sum_vals = np.sum(values)
        if sum_vals > 0:
            win_rate = float(values[1] / sum_vals)
        else:
            win_rate = 0.0
            
        n_wins = int(round(win_rate * n_samples))
        n_losses = n_samples - n_wins
        win_rate_pct = round(win_rate * 100, 2)
        
        return [{
            "rule": " AND ".join(current_rule) if current_rule else "All Trades",
            "samples": n_samples,
            "wins": n_wins,
            "losses": n_losses,
            "win_rate_pct": win_rate_pct
        }]

    rules = []
    feature_idx = tree.feature[node_id]
    feature_name = feature_names[feature_idx].replace("pointer_", "")
    
    # Left branch: feature is <= 0.5 (i.e. False)
    left_rule = current_rule + [f"{feature_name} = False"]
    rules.extend(extract_rules_from_tree(tree, feature_names, left_child, left_rule))
    
    # Right branch: feature is > 0.5 (i.e. True)
    right_rule = current_rule + [f"{feature_name} = True"]
    rules.extend(extract_rules_from_tree(tree, feature_names, right_child, right_rule))
    
    return rules


def run_optimization(file_path: str, min_samples_leaf: int, max_rules: int) -> dict:
    logger.info(f"Loading backtest signals from {file_path}...")
    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError("Signals file is empty.")
        
    # We only optimize on closed signals that hit target, stop loss, or timeout
    closed_df = df[df["outcome"].isin(["target_hit", "stop_loss_hit", "timeout"])].copy()
    if len(closed_df) < 30:
        raise ValueError(f"Too few closed signals ({len(closed_df)}) to run machine learning optimization. Run backtest on a larger universe/period first.")
        
    # Create target variable: 1 if return is positive, 0 otherwise
    closed_df["target_win"] = (closed_df["return_pct"] > 0).astype(int)
    
    # Feature selection
    pointer_cols = [c for c in closed_df.columns if c.startswith("pointer_")]
    if not pointer_cols:
        raise ValueError("No pointer_ columns found in the CSV. Make sure you are using the updated backtest script.")
        
    X = closed_df[pointer_cols].fillna(False).astype(int)
    y = closed_df["target_win"]
    
    # Baseline metrics
    base_samples = len(y)
    base_wins = int(y.sum())
    base_win_rate = round(100.0 * base_wins / base_samples, 2)
    
    logger.info(f"Baseline signals: {base_samples} trades, Win Rate: {base_win_rate}%")
    
    # 1. Feature Importance with Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    feature_importances = []
    for col, imp in zip(pointer_cols, importances):
        name = col.replace("pointer_", "")
        feature_importances.append({
            "feature": name,
            "importance": round(float(imp), 4)
        })
    # Sort by importance descending
    feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)
    
    # 2. Decision Tree Rule Extraction
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=min_samples_leaf, random_state=42)
    dt.fit(X, y)
    
    all_rules = extract_rules_from_tree(dt.tree_, pointer_cols)
    
    # Filter rules: we only want rules that have win rate (greater than baseline) and contain positive outcomes
    # Sort rules by win rate and sample size
    filtered_rules = [r for r in all_rules if r["win_rate_pct"] > base_win_rate and r["samples"] >= min_samples_leaf]
    filtered_rules = sorted(filtered_rules, key=lambda x: (x["win_rate_pct"], x["samples"]), reverse=True)[:max_rules]
    
    # 3. Find the best individual stacked combo (greedy search for score threshold or top combo)
    # Let's check how performance changes with score threshold
    score_analysis = []
    for s in sorted(closed_df["score"].unique()):
        subset = closed_df[closed_df["score"] >= s]
        if len(subset) >= 10:
            wins = (subset["return_pct"] > 0).sum()
            total = len(subset)
            avg_ret = subset["return_pct"].mean()
            
            # calculate profit factor
            gross_gain = subset.loc[subset["return_pct"] > 0, "return_pct"].sum()
            gross_loss = -subset.loc[subset["return_pct"] < 0, "return_pct"].sum()
            pf = gross_gain / gross_loss if gross_loss > 0 else 99.9
            
            score_analysis.append({
                "score_threshold": int(s),
                "samples": int(total),
                "win_rate_pct": round(100.0 * wins / total, 2),
                "avg_return_pct": round(float(avg_ret), 2),
                "profit_factor": round(float(pf), 2)
            })
            
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": os.path.basename(file_path),
        "baseline": {
            "total_trades": base_samples,
            "wins": base_wins,
            "win_rate_pct": base_win_rate
        },
        "feature_importances": feature_importances,
        "extracted_rules": filtered_rules,
        "score_thresholds": score_analysis
    }
    
    return report


def main():
    args = parse_args()
    file_path = args.file
    
    if not file_path:
        file_path = get_latest_signals_file()
        if not file_path:
            logger.error("No backtest signals file found in backtest_results/. Please run a backtest first.")
            return 1
            
    try:
        report = run_optimization(file_path, args.min_samples, args.max_rules)
        
        # Save report
        os.makedirs("backtest_results", exist_ok=True)
        report_path = "backtest_results/optimized_rules.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Optimization report saved to: {report_path}")
        
        # Print summary report
        print("\n" + "=" * 60)
        print("                 SWING STRATEGY OPTIMIZER REPORT")
        print("=" * 60)
        print(f"Source file        : {report['source_file']}")
        print(f"Total trades analyzed: {report['baseline']['total_trades']}")
        print(f"Baseline win rate   : {report['baseline']['win_rate_pct']}%")
        print("-" * 60)
        print("Top 5 Predictive Indicators (Random Forest Importance):")
        for idx, fi in enumerate(report["feature_importances"][:5], 1):
            print(f"  {idx}. {fi['feature']:<25}: {fi['importance'] * 100:.2f}% importance")
            
        print("-" * 60)
        print("Extracted High-Win-Rate Rules (Decision Tree):")
        if not report["extracted_rules"]:
            print("  No rules outperformed the baseline with the minimum sample size constraint.")
        for idx, rule in enumerate(report["extracted_rules"], 1):
            print(f"  Rule {idx}: {rule['rule']}")
            print(f"    - Win Rate: {rule['win_rate_pct']}% ({rule['wins']}/{rule['samples']} wins)")
            
        print("-" * 60)
        print("Score Threshold Optimization:")
        for sa in report["score_thresholds"]:
            print(f"  Score >= {sa['score_threshold']}: {sa['samples']:<4} trades | Win Rate: {sa['win_rate_pct']}% | PF: {sa['profit_factor']:.2f} | Avg Ret: {sa['avg_return_pct']:+.2f}%")
        print("=" * 60 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())