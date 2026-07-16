# NVDA American Option Pricing: Lattice vs XGBoost

This project prices American-style NVDA equity options (2021–2022) two ways — a
model-based binomial lattice that explicitly handles early exercise, and a
data-driven XGBoost regressor trained on observed option quotes — and compares
both against real market mid-prices to see whether structure or flexibility
wins, and where each one breaks.

---

## Headline Results

Test set: 64,747 option quotes, 2022‑07‑01 → 2022‑12‑30 (chronological
holdout, never seen during training), after the liquidity filters described
below.

| Model         | MAE ($) | RMSE ($) |
|---------------|--------:|---------:|
| Lattice       |  1.7825 |   3.1391 |
| XGB Baseline  |  4.6570 |   7.3253 |
| XGB Augmented |  2.7976 |   4.7059 |

MAPE is de-emphasized in this table: many test contracts are cheap, far-OTM
options with mid-prices near $0, so small absolute errors can translate into
large percentage errors that don't reflect real pricing quality. For
reference, with liquidity filters applied MAPE is now Lattice 29.24%, XGB
Baseline 153.31%, XGB Augmented 96.68% — still large, but far less extreme
than before filtering (see Known Issues).

**Note the ordering:** the raw lattice price alone still beats both XGBoost
variants on this test set. Adding liquidity filters (below) closed most of
the gap between XGB Augmented and the lattice, but did not close it — see
[Known Issues](#known-issues) for the full before/after comparison; this
remains an open, actively-investigated regression, not a typo.

---

## Figures

![Error vs Moneyness](figures/error_vs_moneyness.png)

![Prediction Error Distribution](figures/error_histograms.png)

![IV Ablation Comparison](figures/iv_ablation_comparison.png)

---

## Methodology

- **Data**: NVDA American equity option chains, 2021–2022, sourced from
  [OptionsDX](https://www.optionsdx.com/). Cleaned to quote date, strike,
  expiration, bid/ask, volume, and implied vol (`data_cleaner.py`).
- **Train/test split is chronological**, not random: train = quotes before
  2022‑07‑01 (235,253 rows), test = quotes on/after 2022‑07‑01 (64,747 rows).
  This avoids look-ahead bias — the model never trains on data from after the
  date it's evaluated on.
- **Liquidity filters** (`src/features.py`, applied post-melt, alongside the
  existing bad-quote removal): `DTE > 2` days, `bid_ask_spread / mid < 0.5`,
  `|log_moneyness| < 1.0`. Together these trim the ~1.42M-row melted universe
  to ~1.16M rows (‑18.4%) before the 300k lattice sample is drawn — see Known
  Issues for why this mattered.
- **Lattice**: American binomial tree, `N=100` steps, priced on a fixed
  300,000-row sample (150k calls / 150k puts, sampled 2021‑01‑04 →
  2022‑12‑30) rather than the full cleaned dataset, for tractable runtime.
- **Risk-free rate**: flat `r = 0.04` proxy for the full 2021–2022 window —
  not a term structure, not fit per-date.
- **Volatility**: 30-day rolling historical volatility by default
  (`hist_vol`), computed from daily NVDA closes with the 2021‑07‑20 4-for-1
  split's artificial return masked out (see Key Findings).
- **Dividends**: not modeled. NVDA pays a small dividend; the lattice and BAW
  pricers here assume a non-dividend underlier.

---

## Key Findings

- **Split-vol contamination bug (fixed).** NVDA's 2021‑07‑20 4-for-1 split
  produces a ~‑140% one-day "return" in the raw underlying series. Left
  unmasked, this single day poisons every 30-day rolling historical-vol
  window that contains it, inflating `hist_vol` for weeks around the split.
  `src/features.py` now masks `|log_ret| >= 0.5` before computing the rolling
  window.
- **IV ablation confirms the error is mostly a volatility-input problem, not
  a pricing-model problem.** Re-pricing 282,105 valid-IV rows (94.0% of the
  filtered 300k sample) with market-implied vol instead of historical vol
  more than **halves** lattice error: hist-vol lattice MAE $5.0273 / RMSE
  $9.7318 vs IV-lattice MAE $2.1117 / RMSE $4.1770, both measured against
  market mid on the same rows. The lattice math is sound; the historical-vol
  *estimate* feeding it is the weaker link. This conclusion is unchanged
  from before the liquidity filters were added — if anything the gap is
  slightly wider now. See `src/iv_ablation.py` and
  `figures/iv_ablation_comparison.png`.
- **Put-call parity holds almost exactly inside the lattice** (self-consistency
  check, independent of market data): 98.93% of 35,144 matched call/put pairs
  fall within `[S-K, S-K·e^-rT]` using lattice prices, and the remaining
  1.07% "violations" are ~1e-12 in magnitude — floating-point noise, not real
  breaks. **Market mid-prices satisfy the same bound only 76.82% of the
  time** (23.18% violations, avg magnitude $1.05, max $564.31) — expected,
  since the theoretical bound assumes no dividend and NVDA pays one, and
  because market quotes include bid/ask noise. Both figures improved after
  the liquidity filters (previously 96.33% lattice / 70.39% market, on
  31,127 pairs) — the filters removed a chunk of the market-side violations
  along with their average magnitude ($5.13 → $1.05). The single worst
  violation ($564) still survives: it's the (strike=$470, T=3 days) pair
  quoted on **2021‑07‑20 — NVDA's split date itself** (`mid_call=$280.63`
  against `mid_put=$0.035` and an underlying of $186.13), i.e. stale/
  unadjusted pricing during the split transition. It passes the
  `|log_moneyness|` filter at `-0.93` — just inside the ±1.0 cutoff — so
  none of the three filters catch it. See `src/parity_check.py` and
  `figures/put_call_parity_residuals.png`.
- **Lattice sample scaled to 300,000 rows** (150k calls / 150k puts) for this
  round of analysis, up from a smaller pilot sample used during initial
  development — full re-run comparison numbers from that earlier, smaller
  run were not preserved, so only the current 300k-row results are reported
  here.
- **Adding the missing liquidity filters (below) substantially narrowed, but
  did not close, the XGB Augmented regression.** See Known Issues for the
  full before/after comparison.

---

## Known Issues

**XGB Augmented still underperforms the raw lattice price it's supposed to
improve on — but liquidity filters closed most of the gap.**

`src/features.py` was missing the liquidity filters this project was meant
to enforce — only basic bad-quote removal (`mid > 0` and not
`(volume == 0 and bid == 0)`) was implemented. Three filters were added
post-melt, alongside that check: `DTE > 2` days, `bid_ask_spread / mid <
0.5`, and `|log_moneyness| < 1.0`.

**Before vs after**, same modeling code, same 300k-row lattice sample size
and `N=100`, different (filtered) input population:

| Model         | MAE before | MAE after | RMSE before | RMSE after | MAPE before | MAPE after |
|---------------|-----------:|----------:|-------------:|-----------:|-------------:|-----------:|
| Lattice       |     1.5308 |    1.7825 |       2.8696 |     3.1391 |       35.46% |     29.24% |
| XGB Baseline  |     5.3217 |    4.6570 |       9.2325 |     7.3253 |      346.78% |    153.31% |
| XGB Augmented |     3.9411 |    2.7976 |       8.5514 |     4.7059 |      421.48% |     96.68% |

XGB Augmented's RMSE nearly halved (8.55 → 4.71) and its MAPE dropped from
421% to 97%. XGB Baseline improved similarly. The lattice's own MAE/RMSE
ticked up slightly — the filtered population removed a lot of easy,
near-worthless far-OTM contracts the lattice priced trivially well, so the
remaining population is, in dollar terms, a harder mix — but the lattice's
MAPE still improved (35.46% → 29.24%), consistent with the filters removing
noisy small-denominator rows rather than making pricing genuinely worse.

**Verified fixed:** the diagnostic was re-run against the filtered 300k-row
sample. The specific artifact — deep-ITM puts at legacy pre-split $600
strikes — is gone, not just reduced:

- Min DTE: 1.0 → **3.0** days (filter is `T > 2/365`, strictly, so DTE ≥ 3
  survives, not DTE ≥ 2 as the code comment says).
- Max `|log_moneyness|`: 3.65 → **1.00** (filter boundary, as expected).
- Max `bid_ask_spread / mid`: 2.0 → **0.50** (3 rows sit exactly at the
  0.50 boundary — the filter divides by `mid + 1e-6` to avoid a div-by-zero,
  so a handful of rows land a hair under the filter's threshold but exactly
  at 0.50 when recomputed without the epsilon; this is floating-point
  boundary noise, not a filter failure).
- Worst single test-set error dropped from $214.27 (a $600-strike put) to
  **$115.27** (a 3-day, near-the-money put) — no $600-strike names appear
  anywhere in the new worst-20 list.

**Not yet fixed:** the underlying ITM/long-dated tail bias persists at
roughly the same *proportions* even with the legacy-strike artifact gone.
Of the worst 1% of test errors (648 rows, down from 670): **91.5% are ITM**
and **48.1% have >180 days to expiry** — nearly identical to the pre-filter
91.9% / 49.0%. So filtering fixed one specific data artifact, but there's a
second, more structural pattern — XGBoost (with or without the lattice
feature) still struggles disproportionately on long-dated, deep-ITM
contracts — that these filters don't address.

**Leading theories** for the remaining gap (not yet confirmed):

1. Long-dated deep-ITM contracts, even legitimate ones, are a small and
   arguably still-noisy slice of the training distribution (thin volume,
   wide spreads relative to a big intrinsic value) — XGBoost may need more
   ITM/long-dated representation, feature engineering (e.g. intrinsic value
   as an explicit feature), or a separate model for that regime.
2. `src/xgb_model.py` currently trains both models with
   `early_stopping_rounds=30` against `eval_set=[(X_test, y_test)]` — i.e.
   the test set itself picks the best boosting round. This isn't full label
   leakage, but it is a form of test-set-informed model selection worth
   revisiting.

This is being tracked as a separate, ongoing investigation — `src/xgb_model.py`
was intentionally not modified in this pass.

---

## Limitations

- Lattice pricing here uses flat historical volatility by default, not
  implied vol — see the IV ablation above for how much this costs in
  accuracy.
- No dividend modeling anywhere in the pipeline; NVDA pays a small dividend.
- Single ticker (NVDA) — findings may not generalize to other names,
  especially ones without a stock split in the sample window.
- The 2021–2022 window contains NVDA's 4-for-1 split (2021‑07‑20) plus a
  sharp 2022 bear-market drawdown — an unusually volatile regime, not
  necessarily representative of "typical" market conditions.
- The lattice is priced on a fixed 300,000-row sample, not the full cleaned
  dataset, for runtime reasons.

---

## Reproduction

1. Get the raw data from [OptionsDX](https://www.optionsdx.com/) (paid,
   licensed — **not redistributed in this repo**; see `data/README.md`).
2. `python data_cleaner.py` to produce `nvda_cleaned_2021_2022.csv`.
3. `pip install -r requirements.txt`
4. `make all` — runs features → lattice pricing → XGBoost training →
   evaluation figures.
   - `make smoke` — quick binomial/trinomial/Black-Scholes sanity check.
   - `python src/iv_ablation.py` — IV ablation (Task 1 above).
   - `python src/parity_check.py` — put-call parity check (Task 2 above).

---

## Repository Structure

```text
.
├── binomial.py                    # binomial/trinomial/BAW American pricers
├── data_cleaner.py                # raw OptionsDX CSV -> cleaned 2021-2022 CSV
├── Makefile
├── requirements.txt
├── src/
│   ├── features.py                # feature engineering, hist_vol, split-day fix
│   ├── lattice_pricer.py          # prices a sample with the lattice, saves CSV
│   ├── xgb_model.py                # trains baseline/augmented XGBoost
│   ├── evaluation.py               # generates the 5 core comparison figures
│   ├── iv_ablation.py              # re-prices with IV instead of hist_vol
│   └── parity_check.py             # put-call parity no-arbitrage check
├── data/
│   ├── README.md                   # how to obtain/regenerate the data (not tracked)
│   ├── nvda_with_lattice.csv       # gitignored — regenerate via make
│   ├── nvda_test_predictions.csv   # gitignored — regenerate via make
│   └── nvda_lattice_iv_ablation.csv# gitignored — regenerate via iv_ablation.py
├── models/
│   └── xgb_augmented.json          # tracked (< 1 MB)
└── figures/
    ├── error_vs_moneyness.png
    ├── error_vs_maturity.png
    ├── error_histograms.png
    ├── feature_importance.png
    ├── scatter_lattice_vs_market.png
    ├── iv_ablation_comparison.png
    └── put_call_parity_residuals.png
```
