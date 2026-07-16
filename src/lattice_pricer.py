from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MAX_WORKERS = min(os.cpu_count() or 4, 8)

from binomial import binomial_american, trinomial_american, baw_american  # noqa: E402
from src.features import load_features                                      # noqa: E402


def _worker(args):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    row_dict, N, model = args
    try:
        S0 = float(row_dict['underlying'])
        K  = float(row_dict['strike'])
        r  = float(row_dict['r'])
        T  = float(row_dict['T'])
        raw_vol = row_dict.get('hist_vol')
        if raw_vol is None or (isinstance(raw_vol, float) and np.isnan(raw_vol)):
            sigma = 0.01
        else:
            sigma = max(float(raw_vol), 0.01)
        opttype = 'C' if row_dict['opttype_encoded'] == 0 else 'P'
        if model == 'binomial':
            return binomial_american(S0, K, r, sigma, T, N, opttype=opttype)
        elif model == 'trinomial':
            return trinomial_american(S0, K, r, sigma, T, N, opttype=opttype)
        else:  # 'baw'
            return baw_american(S0, K, r, sigma, T, opttype=opttype)
    except Exception:
        return float('nan')


def price_row(row, N: int = 50, model: str = 'binomial') -> float:
    return _worker((row.to_dict(), N, model))


def price_dataframe(df: pd.DataFrame, N: int = 50, model: str = 'binomial') -> pd.Series:
    args = [(row.to_dict(), N, model) for _, row in df.iterrows()]
    chunksize = max(50, len(args) // 200)
    results = []
    print(f"[lattice] Pricing {len(args):,} rows  model={model}  N={N}  "
          f"chunksize={chunksize}  max_workers={MAX_WORKERS} ...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for price in tqdm(
            executor.map(_worker, args, chunksize=chunksize),
            total=len(args),
            desc="Pricing",
        ):
            results.append(price)
    return pd.Series(results, index=df.index)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Price NVDA options with a lattice model.")
    parser.add_argument('--model',  default='binomial',
                        choices=['binomial', 'trinomial', 'baw'],
                        help='Pricing model (default: binomial)')
    parser.add_argument('--n',      type=int, default=50,
                        help='Lattice steps — ignored for baw (default: 50)')
    parser.add_argument('--sample', type=int, default=150_000,
                        help='Rows to price; 0 = all (default: 150000)')
    args = parser.parse_args()

    np.random.seed(42)
    data_dir = ROOT / 'data'
    data_dir.mkdir(exist_ok=True)

    print("[lattice] Loading features ...")
    df = load_features(ROOT / 'nvda_cleaned_2021_2022.csv')

    # Stratified sample (balanced calls/puts), or full dataset
    if args.sample == 0 or args.sample >= len(df):
        sample = df.copy().reset_index(drop=True)
        print(f"[lattice] Using full dataset: {len(sample):,} rows")
    else:
        calls = df[df['opttype_encoded'] == 0]
        puts  = df[df['opttype_encoded'] == 1]
        half  = args.sample // 2
        n_calls = min(half, len(calls))
        n_puts  = min(args.sample - n_calls, len(puts))
        sample = pd.concat([
            calls.sample(n_calls, random_state=42),
            puts.sample(n_puts,   random_state=42),
        ]).reset_index(drop=True)
        print(f"[lattice] Sample: {len(sample):,} rows  "
              f"({n_calls:,} calls, {n_puts:,} puts)")

    prices = price_dataframe(sample, N=args.n, model=args.model)
    sample['lattice_price'] = prices.values

    out_path = data_dir / 'nvda_with_lattice.csv'
    sample.to_csv(out_path, index=False)
    print(f"[lattice] Saved → {out_path}")
    print(sample.head(5).to_string())
    nan_count = sample['lattice_price'].isna().sum()
    print(f"[lattice] NaN prices: {nan_count:,} / {len(sample):,}")
