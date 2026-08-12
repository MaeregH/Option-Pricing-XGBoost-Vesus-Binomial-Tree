# NVDA American Option Pricing: Lattice vs XGBoost

This project prices American-style NVDA equity options (2021–2022) two ways — a
model-based binomial lattice that explicitly handles early exercise, and a
data-driven XGBoost regressor trained on observed option quotes — and compares
both against real market mid-prices to see whether structure or flexibility
wins, and where each one breaks.

---

## Headline Results

Test set: 64,747 option quotes, 2022‑07‑01 → 2022‑12‑30 (chronological
holdout, never seen during training or validation), after the liquidity
filters described below and using a forward-chaining train/validation split
(see Methodology and Key Findings).

| Model         | MAE ($) | RMSE ($) |
|---------------|--------:|---------:|
| Lattice       |  1.7825 |   3.1391 |
| XGB Baseline  |  6.1857 |   9.7840 |
| XGB Augmented |  3.1421 |   6.0060 |

MAPE is de-emphasized in this table: many test contracts are cheap, far-OTM
options with mid-prices near $0, so small absolute errors can translate into
large percentage errors that don't reflect real pricing quality. For
reference, MAPE is Lattice 29.24%, XGB Baseline 119.83%, XGB Augmented
138.32% (Augmented's MAPE being worse than Baseline's here, despite a clearly
better MAE/RMSE, is exactly this de-emphasis effect in action — a handful of
badly-missed cheap contracts dominate MAPE; see Known Issues for the specific
tail driving this).

**Note the ordering:** the raw lattice price alone still beats both XGBoost
variants on this test set in absolute dollar terms. XGB Augmented does
clearly and consistently beat XGB Baseline (RMSE $6.01 vs $9.78, ~39% lower)
— that comparison is now trustworthy after fixing a test-set leakage bug in
early stopping (see Key Findings). The remaining Augmented-vs-Lattice gap is
a real, still-open finding — a structural ITM/long-dated pricing bias, not a
methodology bug — tracked in [Known Issues](#known-issues).

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
- **Train/validation/test split is chronological and forward-chaining**, not
  random, and validation is not a random slice of training history: train =
  quotes before 2022‑06‑01 (224,316 rows), validation = quotes in June 2022
  only (10,937 rows, used exclusively for early stopping — `xgb_model.py`'s
  `early_stopping_rounds=30`), test = quotes on/after 2022‑07‑01 (64,747
  rows, fully held out until final evaluation). Validation sits immediately
  before test in time so early stopping sees data from the same regime it
  will be evaluated near — see Key Findings for why this specific choice
  (over a random or distant-past validation slice) matters a great deal here.
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

- **Test-set leakage in early stopping (found and fixed) — XGBoost's real
  held-out performance was hidden until this was resolved.**
  `early_stopping_rounds=30` was originally run against
  `eval_set=[(X_test, y_test)]` — the boosting round was being chosen using
  test-set labels, inflating the reported XGB numbers. Fixing this took two
  attempts:

  | Split methodology | Train ends | Validation | Baseline MAE/RMSE | Augmented MAE/RMSE |
  |---|---|---|---:|---:|
  | Leaky (eval_set = test itself) | 2022-06-30 | none (test) | 4.66 / 7.33 | 2.80 / 4.71 |
  | Attempt 1: 15%-of-training carve-out | 2022-04-04 | Apr 5 – Jun 30 (35,536 rows) | 17.40 / 27.44 | 11.23 / 27.60 |
  | **Attempt 2: forward-chaining (adopted)** | **2022-05-31** | **Jun 1–30 (10,937 rows)** | **6.19 / 9.78** | **3.14 / 6.01** |

  The first, naive fix (a 15%-of-training date carve-out) removed the
  leakage but overcorrected: cutting training off 3 months before the test
  window starved early stopping of the most test-adjacent, regime-relevant
  data. RMSE cratered (both models got *worse* than the leaky baseline), and
  early stopping quit at round 90/500 — the April–June 2022 validation slice
  is different enough in volatility regime from most of 2021 training data
  that validation loss plateaued artificially early. Controlled experiments
  (shrinking the carve-out to 5%, then switching to forward-chaining —
  train through 2022‑05‑31, validate on June 2022 exactly) confirmed
  **recency, not validation-set size, was the driver**: with forward-chaining
  validation, early stopping now runs to round 486–496 of 500 (close to
  budget, not truncating early), and RMSE recovers to within a few dollars of
  the leaky ceiling. XGB Augmented's advantage over Baseline on RMSE, which
  had briefly gone *negative* under the 15% carve-out, is restored and
  slightly exceeds what it appeared to be under the original leaky setup
  (38.6% lower RMSE vs. baseline, vs. 35.8% under the leaky methodology).
  This is now the adopted methodology in `src/xgb_model.py` and the headline
  numbers above.

  One reproducibility note surfaced while validating this fix: with
  `subsample=0.8`, XGBoost samples rows *by position*, not content, so the
  input row order affects results even for identical data — the source CSV's
  row order (from a stratified call/put sample, not chronological) shifted
  Augmented's RMSE by ~10% run-to-run before `xgb_model.py` was changed to
  sort by `quote_date` prior to splitting. A small residual gap remains even
  after sorting, consistent with XGBoost's documented non-determinism under
  multi-threaded histogram construction. This doesn't change the conclusion
  above — it's a few-percent effect on top of a >4x recovery — but it's why
  the exact decimals here may not reproduce bit-for-bit on a re-run.
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
- **Adding the missing liquidity filters (below) substantially narrowed the
  gap between XGB Augmented and the lattice**, before the leakage finding
  above further changed the picture. See Known Issues for the filter
  before/after comparison (measured under the split methodology in place at
  that time).

---

## Known Issues

**XGB Augmented still underperforms the raw lattice price it's supposed to
improve on, concentrated in a specific ITM/long-dated tail** — liquidity
filters and, separately, a validation-methodology fix (Key Findings) each
narrowed the overall gap, but neither touched this specific structural
pattern.

`src/features.py` was originally missing the liquidity filters this project
was meant to enforce — only basic bad-quote removal (`mid > 0` and not
`(volume == 0 and bid == 0)`) was implemented. Three filters were added
post-melt, alongside that check: `DTE > 2` days, `bid_ask_spread / mid <
0.5`, and `|log_moneyness| < 1.0`.

**Before vs after** (historical record, measured under the leaky
`eval_set=test` split methodology in place at the time, before the
validation fix in Key Findings — held constant here to isolate the filters'
effect on the data population), same 300k-row lattice sample size and
`N=100`, different (filtered) input population:

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
roughly the same *proportions* regardless of the legacy-strike artifact fix
or the validation-methodology fix (Key Findings) — if anything it's slightly
more concentrated now that early stopping is actually working correctly. Of
the worst 1% of test errors under the adopted forward-chaining split (648
rows, threshold $30.58): **88.9% are ITM** and **57.3% have >180 days to
expiry**. This is essentially the same shape seen under every split
methodology tried so far (leaky, 15%-carve-out, forward-chaining) — strong
evidence this is a genuine model/data limitation, not an artifact of the
leakage bug or its fix.

**Is the tail actually mispriced, or just expensive?** Computed
`abs_error / mid` (relative error) for every test row (measured against the
final, adopted forward-chaining model's predictions) and compared the worst
1% (n=648, threshold $30.58) against the remaining 99% (n=64,099):

| Segment | mean abs_error | median abs_error | mean mid ($) | median mid ($) | mean rel_error | median rel_error |
|---|---:|---:|---:|---:|---:|---:|
| Worst 1% (tail) | $39.62 | $38.08 | $185.42 | $195.51 | 141.3% | **19.62%** |
| Remaining 99%   | $2.77  | $1.79  | $45.93  | $22.60  | 296.7%¹ | 8.71% |

¹ The 99%'s *mean* relative error is inflated by near-zero-`mid` far-OTM
contracts where a few cents of dollar error becomes a triple-digit percentage
— the same MAPE-explosion effect already visible in the headline metrics.
Medians are the honest comparison here.

**Answer: both effects are real, and the mispricing gap is, if anything,
slightly wider now that early stopping is fixed.** Dollar scale still
explains part of the gap — tail options are ~8.6x more expensive than the
typical test row (median $195.51 vs $22.60). But it isn't purely mechanical:
restricting to ITM options only (removing the "expensive vs cheap"
confound), the tail's median relative error is **18.81%** (n=576) vs
**4.00%** for ITM options in the rest of the test set (n=34,651) — a
**~4.7x** gap that survives controlling for moneyness (up from ~4.4x under
the row-order-unsorted forward-chaining run, and ~3.2x under the original
leaky model). The lattice-augmented model really is pricing these contracts
proportionally worse, not just paying a dollar-scale penalty for their size
— and fixing the leakage bug made this *more* visible, not less, which is
itself evidence the effect is real rather than a leakage artifact.

DTE still does not look like an independent driver once inside the tail:
median relative error is nearly identical for >180-DTE tail rows (19.48%,
n=371) and ≤180-DTE tail rows (19.97%, n=277). Long-dated and deep-ITM tend
to co-occur (far-ITM contracts accumulate intrinsic value, and intrinsic
value grows with time), so the >180-DTE figure looks like it's riding along
with the ITM effect rather than contributing its own separate failure mode.

One further wrinkle, also still present and now larger: 72 of the 648 tail
rows (11.1%, up from 8.6%) are **not** ITM, and they are a distinct, worse
failure — median relative error **985.8%**, driven by far-OTM, long-dated
contracts (DTE 318–634 days) that the market prices at $3–8 in pure time
value but the model predicts at $65–81. Example: strike $335 vs underlying
$143.02 (call, DTE 592), market mid $5.23, model prediction $81.34. These
pass the `|log_moneyness| < 1.0` filter but appear to sit in a sparse,
poorly-generalized region of the long-dated/far-OTM feature space — a
candidate second, smaller failure mode distinct from the deep-ITM story
above.

**Leading theory** for the remaining gap (not yet confirmed): long-dated
deep-ITM contracts, even legitimate ones, are a small and arguably
still-noisy slice of the training distribution (thin volume, wide spreads
relative to a big intrinsic value) — XGBoost may need more ITM/long-dated
representation, feature engineering (e.g. intrinsic value as an explicit
feature), or a separate model for that regime. (A second theory — test-set
leakage in early stopping — was investigated and resolved; see Key Findings.
It was not the cause of this particular tail bias: the bias's shape and
magnitude are stable, or slightly worse, across every split methodology
tested, including the fully corrected one.)

This is being tracked as a separate, ongoing investigation.

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
