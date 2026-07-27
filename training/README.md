# BTST Parameter Optimization Methodology

This directory contains the code to optimize and validate parameters for the BTST trading strategy using a rigorous, non-overlapping train/validation/holdout split, walk-forward validation folds, and statistical significance testing.

## Data Partitioning & Leakage Prevention

The data is strictly partitioned into two main phases to prevent selection bias and ensure reported performance is a genuine out-of-sample estimate:

1. **Optimization Phase (2021-07-22 to 2026-01-22):**
   This period is used for parameter search. It is split into **5 rolling walk-forward folds** (each with non-overlapping Train and Validation segments separated by a **4-day purged gap** to avoid leakage across boundary days).
2. **Holdout Test Phase (2026-01-23 to 2026-07-22):**
   This period is completely locked during grid-search parameter selection. The holdout dataset is loaded and evaluated **exactly once** at the very end of the run using the locked configuration.

### Rolling Walk-forward Fold Structure

Within the optimization period, the script steps forward chronologically over 5 validation folds:
* **Train Segment:** 2 years of daily data per fold.
* **Purge Gap:** 4 calendar days (at least 2 trading days) to ensure no forward-looking features leak across the train/validation boundary.
* **Validation Segment:** 6 months of daily data per fold.

---

## Statistical Significance & Execution Gating

To prevent overfitting on small sample sizes, candidate configurations must pass multiple validation gates:

1. **Binomial Significance Test:**
   A one-sided binomial test gates the validation results. The probability of obtaining the observed wins under a null hypothesis ($p_{null} = 50\%$) is computed:
   $$p = \sum_{k=\text{wins}}^{N} \binom{N}{k} (0.50)^k (0.50)^{N-k}$$
   We require $p \le 0.05$ (configurable via `--p-value-threshold`) in a majority of folds.
2. **Circuit Limit Filter:**
   Signals are excluded if the close price is near the day's high (`(High - Close) / Close <= 0.05%`) and the daily range is very narrow (`(High - Low) / Close < 0.5%`), since executions cannot be completed when a circuit lock occurs.
3. **Liquidity/Turnover Filter:**
   Signals are only accepted if the 20-day average daily turnover (Close × Volume) is $\ge$ 1 Crore.
4. **Majority Pass Criteria:**
   A configuration is only valid if it meets the minimum trade counts, is net profitable, and is statistically significant in a majority of the 5 folds.

---

## Optimization Modes

Run the script with the `--mode` flag:
* `pooled` (Default): Fits a single shared parameter configuration across all stocks pooled together. Highly recommended to prevent stock-specific overfitting.
* `hierarchical`: Falls back to the global pooled parameters unless a stock has $\ge$ 30 validation trades and shows a statistically significant improvement with custom weights.
* `per-stock`: Optimizes stock-by-stock independently (forces walk-forward significance tests per stock).

---

## How to Roll Forward the Holdout Period

As new market data arrives, you should roll the date boundaries forward to prevent reusing the already-seen holdout set for parameter tweaking:

1. **Shift Dates:** In `train_btst_model.py` `main()`, shift the dates forward (e.g., slide the holdout start and end dates by 6 months).
2. **Clear Prior Configs:** Always delete/backup the prior `btst_stock_weights.json` so you do not accidentally reuse parameters tuned on past data.
3. **Run Optimization:** Execute the script to generate fresh parameters and evaluate them against the new holdout period:
   ```bash
   uv run training/train_btst_model.py --mode pooled
   ```
