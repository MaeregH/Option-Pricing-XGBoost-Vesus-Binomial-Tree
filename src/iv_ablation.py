from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.lattice_pricer import price_dataframe  # noqa: E402

DATA = ROOT / 'data'
FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

sns.set_style('whitegrid')
FIGSIZE = (10, 6)
DPI = 150

N_STEPS = 100
MODEL = 'binomial'


def _metrics(y_true, y_pred, name: str):
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    print(f"  {name:28s}  MAE={mae:.4f}  RMSE={rmse:.4f}  (n={mask.sum():,})")
    return mae, rmse


def price_with_iv(df: pd.DataFrame) -> pd.Series:
    """Reprice a copy of df using implied vol in place of hist_vol via the
    existing lattice pricer, leaving df itself untouched."""
    priced = df.copy()
    priced['hist_vol'] = priced['iv'].clip(0.01, 3.0)
    return price_dataframe(priced, N=N_STEPS, model=MODEL)


if __name__ == "__main__":
    np.random.seed(42)

    src_path = DATA / 'nvda_with_lattice.csv'
    print(f"[iv_ablation] Loading {src_path} ...")
    df = pd.read_csv(src_path, parse_dates=['quote_date'])
    print(f"[iv_ablation] Full rows: {len(df):,}")

    valid = df[df['iv'].notna() & (df['iv'] > 0)].copy().reset_index(drop=True)
    print(f"[iv_ablation] Rows with valid IV: {len(valid):,} / {len(df):,} "
          f"({len(valid) / len(df) * 100:.1f}%)")

    # Stage 1: timing check on a 5,000-row subset
    test_n = min(5000, len(valid))
    test_sample = valid.sample(test_n, random_state=42).reset_index(drop=True)
    print(f"[iv_ablation] Stage 1 — timing check on {test_n:,} rows ...")
    t0 = time.time()
    test_prices = price_with_iv(test_sample)
    elapsed = time.time() - t0
    nan_count = test_prices.isna().sum()
    print(f"[iv_ablation] Stage 1 done in {elapsed:.1f}s "
          f"({elapsed / test_n * 1000:.2f} ms/row)  NaN: {nan_count}/{test_n}")
    est_full_min = elapsed / test_n * len(valid) / 60
    print(f"[iv_ablation] Estimated full run: ~{est_full_min:.1f} min for {len(valid):,} rows")

    # Stage 2: full run on all valid-IV rows
    print(f"[iv_ablation] Stage 2 — full run on {len(valid):,} rows ...")
    t0 = time.time()
    iv_prices = price_with_iv(valid)
    elapsed_full = time.time() - t0
    print(f"[iv_ablation] Stage 2 done in {elapsed_full / 60:.1f} min")

    valid['lattice_price_iv'] = iv_prices.values
    nan_count_full = valid['lattice_price_iv'].isna().sum()
    print(f"[iv_ablation] NaN lattice_price_iv: {nan_count_full:,} / {len(valid):,}")

    # Apples-to-apples comparison: same rows, both lattice prices + mid present
    cmp_df = valid.dropna(subset=['lattice_price', 'lattice_price_iv', 'mid']).copy()
    print(f"\n[iv_ablation] Comparison set: {len(cmp_df):,} rows "
          f"(hist-vol lattice, IV lattice, and mid all present)")
    print("=== MAE / RMSE vs market mid ===")
    _metrics(cmp_df['mid'], cmp_df['lattice_price'],    'lattice_price (hist_vol)')
    _metrics(cmp_df['mid'], cmp_df['lattice_price_iv'], 'lattice_price_iv (IV)')

    out_path = DATA / 'nvda_lattice_iv_ablation.csv'
    valid.to_csv(out_path, index=False)
    print(f"[iv_ablation] Saved → {out_path}")

    # Figure: overlaid histograms of (lattice_price - mid) for hist-vol vs IV
    fig, ax = plt.subplots(figsize=FIGSIZE)
    err_hist = cmp_df['lattice_price'] - cmp_df['mid']
    err_iv = cmp_df['lattice_price_iv'] - cmp_df['mid']
    ax.hist(err_hist, bins=80, alpha=0.4, label='Lattice (hist_vol)')
    ax.hist(err_iv, bins=80, alpha=0.4, label='Lattice (IV)')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Lattice Price − Market Mid ($)')
    ax.set_ylabel('Count')
    ax.set_title('IV Ablation: Lattice Pricing Error, Hist-Vol vs Implied Vol')
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURES / 'iv_ablation_comparison.png'
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    print(f"[iv_ablation] Saved → {fig_path}")
