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

FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

sns.set_style('whitegrid')
FIGSIZE = (10, 6)
DPI = 150

MODELS = [
    ('lattice_price',       'Lattice'),
    ('xgb_baseline_pred',   'XGB Baseline'),
    ('xgb_augmented_pred',  'XGB Augmented'),
]


def _save(fig, name: str):
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[eval] Saved → {path}")


def fig1_error_vs_moneyness(df: pd.DataFrame):
    df = df.copy()
    df['bucket'] = pd.qcut(df['log_moneyness'], 10, labels=False, duplicates='drop')
    bucket_mono  = df.groupby('bucket')['log_moneyness'].median()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for col, label in MODELS:
        mae_per_bucket = (df[col] - df['mid']).abs().groupby(df['bucket']).median()
        ax.plot(bucket_mono.loc[mae_per_bucket.index], mae_per_bucket.values,
                marker='o', label=label)
    ax.set_xlabel('Log Moneyness  ln(S/K)')
    ax.set_ylabel('Median Absolute Error ($)')
    ax.set_title('Pricing Error vs Moneyness')
    ax.legend()
    _save(fig, 'error_vs_moneyness.png')


def fig2_error_vs_maturity(df: pd.DataFrame):
    bins   = [0, 7 / 365, 30 / 365, 90 / 365, 180 / 365, np.inf]
    labels = ['<7d', '7–30d', '30–90d', '90–180d', '>180d']
    df = df.copy()
    df['maturity_bin'] = pd.cut(df['T'], bins=bins, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, enc, title in zip(axes, [0, 1], ['Calls', 'Puts']):
        sub = df[df['opttype_encoded'] == enc]
        for col, label in MODELS:
            err = (sub[col] - sub['mid']).abs()
            bucket_err = err.groupby(sub['maturity_bin'], observed=True).median()
            ax.plot(bucket_err.index.astype(str), bucket_err.values,
                    marker='o', label=label)
        ax.set_title(f'Pricing Error vs Maturity ({title})')
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('Median Absolute Error ($)')
        ax.legend()

    fig.tight_layout()
    path = FIGURES / 'error_vs_maturity.png'
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[eval] Saved → {path}")


def fig3_error_histograms(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for col, label in MODELS:
        err = df[col] - df['mid']
        ax.hist(err.dropna(), bins=80, alpha=0.4, label=label)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Prediction Error ($)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Error Distribution')
    ax.legend()
    _save(fig, 'error_histograms.png')


def fig4_feature_importance(model_path: Path):
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.load_model(str(model_path))

    features = [
        'underlying', 'strike', 'T', 'r', 'hist_vol',
        'log_moneyness', 'opttype_encoded', 'bid_ask_spread', 'lattice_price',
    ]
    importances = model.feature_importances_
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh([features[i] for i in idx], importances[idx])
    ax.set_xlabel('Importance')
    ax.set_title('XGBoost Feature Importance')
    _save(fig, 'feature_importance.png')


def fig5_scatter_lattice_vs_market(df: pd.DataFrame):
    np.random.seed(42)
    sub = df.sample(min(3000, len(df)), random_state=42)

    calls = sub[sub['opttype_encoded'] == 0]
    puts  = sub[sub['opttype_encoded'] == 1]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(calls['lattice_price'], calls['mid'],
               alpha=0.3, color='steelblue', s=10, label='Calls')
    ax.scatter(puts['lattice_price'],  puts['mid'],
               alpha=0.3, color='darkorange', s=10, label='Puts')

    lim = max(sub['lattice_price'].max(), sub['mid'].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='y = x')
    ax.set_xlabel('Lattice Price ($)')
    ax.set_ylabel('Market Mid Price ($)')
    ax.set_title('Lattice Price vs Market Price')
    ax.legend()
    _save(fig, 'scatter_lattice_vs_market.png')


if __name__ == "__main__":
    pred_path  = ROOT / 'data' / 'nvda_test_predictions.csv'
    model_path = ROOT / 'models' / 'xgb_augmented.json'

    print(f"[eval] Loading {pred_path} ...")
    df = pd.read_csv(pred_path, parse_dates=['quote_date'])
    df = df.dropna(subset=['lattice_price', 'mid', 'xgb_baseline_pred', 'xgb_augmented_pred'])
    print(f"[eval] Generating figures for {len(df)} test rows ...")

    fig1_error_vs_moneyness(df)
    fig2_error_vs_maturity(df)
    fig3_error_histograms(df)
    fig4_feature_importance(model_path)
    fig5_scatter_lattice_vs_market(df)
    print("[eval] All figures saved.")
