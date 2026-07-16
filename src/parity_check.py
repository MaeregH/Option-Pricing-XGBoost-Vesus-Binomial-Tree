from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / 'data'
FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

sns.set_style('whitegrid')
FIGSIZE = (10, 6)
DPI = 150


def pair_calls_puts(df: pd.DataFrame) -> pd.DataFrame:
    """Pair calls with puts sharing the same quote_date, strike, AND time to
    maturity T. Matching on (quote_date, strike) alone would also merge
    contracts from different expirations that happen to share a strike on
    the same quote date, which would make the parity bounds meaningless."""
    calls = df[df['opttype_encoded'] == 0]
    puts = df[df['opttype_encoded'] == 1]
    pairs = calls.merge(
        puts, on=['quote_date', 'strike', 'T'], suffixes=('_call', '_put')
    )
    return pairs


def bounds(pairs: pd.DataFrame):
    underlying = pairs['underlying_call']
    strike = pairs['strike']
    r = pairs['r_call']
    T = pairs['T']
    lower = underlying - strike
    upper = underlying - strike * np.exp(-r * T)
    return lower, upper


def check(parity: pd.Series, lower: pd.Series, upper: pd.Series, label: str):
    mask = parity.notna() & lower.notna() & upper.notna()
    p, lo, hi = parity[mask], lower[mask], upper[mask]

    in_bounds = (p >= lo) & (p <= hi)
    pct_in = in_bounds.mean() * 100

    below = lo - p
    above = p - hi
    violation = pd.concat([below, above], axis=1).max(axis=1).clip(lower=0)
    violated = violation > 0

    print(f"--- {label} ---")
    print(f"  Pairs checked: {mask.sum():,}")
    print(f"  Within [S-K, S-K*e^-rT]: {pct_in:.2f}%")
    if violated.any():
        avg_v = violation[violated].mean()
        max_v = violation[violated].max()
        print(f"  Violations: {violated.sum():,} ({violated.mean() * 100:.2f}%)")
        print(f"  Avg violation magnitude: ${avg_v:.6g}")
        print(f"  Max violation magnitude: ${max_v:.6g}")
        if max_v < 1e-6:
            print("  (magnitude is floating-point noise, i.e. effectively exact)")
    else:
        print("  Violations: 0")
    print()
    return p, lo, hi, violation


if __name__ == "__main__":
    src_path = DATA / 'nvda_with_lattice.csv'
    print(f"[parity] Loading {src_path} ...")
    df = pd.read_csv(src_path, parse_dates=['quote_date'])

    pairs = pair_calls_puts(df)
    print(f"[parity] Matched call/put pairs (same quote_date, strike, T): {len(pairs):,}")
    print("[parity] NOTE: NVDA pays a small dividend that this project does not model.")
    print("[parity] The bounds below assume a non-dividend underlier, so some of the")
    print("[parity] 'violations' reported here are expected dividend effects, not")
    print("[parity] necessarily arbitrage or lattice bugs.\n")

    lower, upper = bounds(pairs)

    print("=" * 70)
    print("MARKET-BASED PUT-CALL PARITY CHECK  (mid_call - mid_put)")
    print("=" * 70)
    parity_mkt = pairs['mid_call'] - pairs['mid_put']
    p_mkt, lo_mkt, hi_mkt, viol_mkt = check(parity_mkt, lower, upper, "Market mid prices")

    print("=" * 70)
    print("LATTICE-BASED PUT-CALL PARITY CHECK  (lattice_price_call - lattice_price_put)")
    print("=" * 70)
    print("[parity] This checks the lattice's internal self-consistency, independent")
    print("[parity] of market data — it should hold almost exactly since the same")
    print("[parity] pricing model/vol/rate feed both legs.\n")
    parity_lat = pairs['lattice_price_call'] - pairs['lattice_price_put']
    check(parity_lat, lower, upper, "Lattice prices")

    # Figure: histogram of market-based residual relative to the no-arbitrage band.
    # Residual = 0 means exactly at the nearer violated bound; negative = inside band
    # (distance to nearest bound is not needed — plot signed distance below lower /
    # above upper, i.e. 0 for in-bounds points is not meaningful, so instead we plot
    # parity - midpoint of [lower, upper] for a single interpretable residual).
    mid_band = (lo_mkt + hi_mkt) / 2
    residual = p_mkt - mid_band

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(residual, bins=80, alpha=0.7, color='steelblue')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Band midpoint')
    ax.set_xlabel('(mid_call − mid_put) − Band Midpoint ($)')
    ax.set_ylabel('Count')
    ax.set_title('Put-Call Parity Residual vs No-Arbitrage Band (Market Prices)')
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURES / 'put_call_parity_residuals.png'
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    print(f"[parity] Saved → {fig_path}")
