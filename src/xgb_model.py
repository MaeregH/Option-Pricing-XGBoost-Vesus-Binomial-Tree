from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BASE_FEATURES = [
    'underlying', 'strike', 'T', 'r', 'hist_vol',
    'log_moneyness', 'opttype_encoded', 'bid_ask_spread',
]
AUG_FEATURES = BASE_FEATURES + ['lattice_price']


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 0.01
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _metrics(y_true, y_pred, name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = _mape(np.asarray(y_true), np.asarray(y_pred))
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}


if __name__ == "__main__":
    np.random.seed(42)

    models_dir = ROOT / 'models'
    models_dir.mkdir(exist_ok=True)

    data_path = ROOT / 'data' / 'nvda_with_lattice.csv'
    print(f"[xgb] Loading {data_path} ...")
    df = pd.read_csv(data_path, parse_dates=['quote_date'])
    df = df.dropna(subset=['lattice_price', 'mid'])
    print(f"[xgb] Clean rows: {len(df)}")

    # Forward-chaining split: validation is the single month immediately
    # before test, so early stopping sees data from the same regime it will
    # be evaluated near — not a random or distant-past validation slice.
    # eval_set was previously the test set itself, letting early stopping
    # pick the best boosting round using test-set labels (leakage); a first
    # fix (a 15%-of-training date carve-out, ending 2022-04-04) removed the
    # leakage but cut early stopping off from the most test-adjacent data,
    # which controlled experiments showed was the dominant effect (RMSE
    # 27.60 at the 15% cut vs 5.00 here) — not validation-set size itself.
    TRAIN_END  = pd.Timestamp('2022-06-01')  # train: everything before this
    TEST_START = pd.Timestamp('2022-07-01')  # test: everything from this on; val fills the gap between

    # Sorted by date: XGBoost's subsample<1.0 samples by row position, so an
    # unsorted (arbitrary) row order changes which rows get subsampled each
    # boosting round even though the underlying data is identical.
    df = df.sort_values('quote_date')
    train = df[df['quote_date'] < TRAIN_END]
    val   = df[(df['quote_date'] >= TRAIN_END) & (df['quote_date'] < TEST_START)]
    test  = df[df['quote_date'] >= TEST_START]
    print(f"[xgb] Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    y_train = train['mid'].values
    y_val   = val['mid'].values
    y_test  = test['mid'].values

    X_train_base = train[BASE_FEATURES].values
    X_val_base   = val[BASE_FEATURES].values
    X_test_base  = test[BASE_FEATURES].values
    X_train_aug  = train[AUG_FEATURES].values
    X_val_aug    = val[AUG_FEATURES].values
    X_test_aug   = test[AUG_FEATURES].values

    common_hp = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=30, eval_metric='rmse',
        random_state=42,
    )

    print("[xgb] Training baseline (no lattice) ...")
    xgb_baseline = XGBRegressor(**common_hp)
    xgb_baseline.fit(
        X_train_base, y_train,
        eval_set=[(X_val_base, y_val)],
        verbose=False,
    )
    print(f"[xgb] Baseline best_iteration: {xgb_baseline.best_iteration} / {common_hp['n_estimators']}")

    print("[xgb] Training augmented (with lattice) ...")
    xgb_augmented = XGBRegressor(**common_hp)
    xgb_augmented.fit(
        X_train_aug, y_train,
        eval_set=[(X_val_aug, y_val)],
        verbose=False,
    )
    print(f"[xgb] Augmented best_iteration: {xgb_augmented.best_iteration} / {common_hp['n_estimators']}")

    lat_pred  = test['lattice_price'].values
    base_pred = xgb_baseline.predict(X_test_base)
    aug_pred  = xgb_augmented.predict(X_test_aug)

    metrics = pd.DataFrame([
        _metrics(y_test, lat_pred,  'Lattice'),
        _metrics(y_test, base_pred, 'XGB Baseline'),
        _metrics(y_test, aug_pred,  'XGB Augmented'),
    ]).set_index('Model')
    print("\n=== Test-set metrics ===")
    print(metrics.to_string(float_format="{:.4f}".format))

    model_path = models_dir / 'xgb_augmented.json'
    xgb_augmented.save_model(str(model_path))
    print(f"\n[xgb] Saved model → {model_path}")

    # Save full test-set predictions for evaluation.py
    test_out = test.copy()
    test_out['xgb_baseline_pred']  = base_pred
    test_out['xgb_augmented_pred'] = aug_pred
    pred_path = ROOT / 'data' / 'nvda_test_predictions.csv'
    test_out.to_csv(pred_path, index=False)
    print(f"[xgb] Saved test predictions → {pred_path}")
