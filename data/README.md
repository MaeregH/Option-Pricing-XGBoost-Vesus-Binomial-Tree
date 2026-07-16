# data/

Raw and derived option-chain data is **not committed to this repo** — the
underlying NVDA option chain is licensed vendor data, and the derived CSVs
regenerated from it run 10–40+ MB each, too large to track sensibly in git.

## What's not included

- `nvda_2020_2022.csv` / `nvda_cleaned_2021_2022.csv` (repo root) — raw and
  cleaned NVDA option chain.
- `data/nvda_with_lattice.csv` — cleaned chain + lattice prices.
- `data/nvda_test_predictions.csv` — test-set rows + XGBoost predictions.
- `data/nvda_lattice_iv_ablation.csv` — IV-ablation lattice repricing.

## Where to get the raw data

Historical NVDA option chains (bid/ask, volume, implied vol, Greeks) can be
purchased from [OptionsDX](https://www.optionsdx.com/). Their data is sold
under a commercial license that does **not** permit redistribution — hence
it isn't checked in here. Download the 2021–2022 NVDA equity option history
and save it as `nvda_2020_2022.csv` in the repo root.

## Regenerating everything

```bash
python data_cleaner.py   # nvda_2020_2022.csv -> nvda_cleaned_2021_2022.csv
make all                 # features -> lattice pricing -> XGBoost -> figures
```

Optional follow-on analyses (require `make all` to have run first):

```bash
python src/iv_ablation.py   # -> data/nvda_lattice_iv_ablation.csv
python src/parity_check.py  # -> figures/put_call_parity_residuals.png
```
