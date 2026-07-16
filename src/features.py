from pathlib import Path
import numpy as np
import pandas as pd

R = 0.04  # fixed risk-free rate proxy


def load_features(path='nvda_cleaned_2021_2022.csv') -> pd.DataFrame:
    path = Path(path)
    print(f"[features] Loading {path} ...")

    df = pd.read_csv(path, skipinitialspace=True)
    print(f"[features] Raw shape: {df.shape}")

    # Parse dates
    df['[QUOTE_DATE]']  = pd.to_datetime(df['[QUOTE_DATE]'])
    df['[EXPIRE_DATE]'] = pd.to_datetime(df['[EXPIRE_DATE]'])

    # Recompute T from DTE
    df['T'] = df['[DTE]'] / 365.0

    # Mid prices
    df['call_mid'] = (df['[C_BID]'] + df['[C_ASK]']) / 2
    df['put_mid']  = (df['[P_BID]'] + df['[P_ASK]']) / 2

    # Log moneyness
    df['log_moneyness'] = np.log(df['[UNDERLYING_LAST]'] / df['[STRIKE]'])

    # 30-day rolling historical volatility computed on unique daily underlying prices
    print("[features] Computing rolling historical volatility ...")
    daily = (
        df[['[QUOTE_DATE]', '[UNDERLYING_LAST]']]
        .drop_duplicates('[QUOTE_DATE]', keep='first')
        .sort_values('[QUOTE_DATE]')
        .reset_index(drop=True)
    )
    daily['log_ret'] = np.log(daily['[UNDERLYING_LAST]'] / daily['[UNDERLYING_LAST]'].shift(1))

    # NVDA's 2021-07-20 4-for-1 split creates a ~-1.4 "return" that isn't a real
    # price move — it blows up any 30-day rolling window that contains it.
    extreme_mask = daily['log_ret'].abs() >= 0.5
    masked_dates = daily.loc[extreme_mask, '[QUOTE_DATE]'].dt.strftime('%Y-%m-%d').tolist()
    daily['log_ret'] = daily['log_ret'].where(daily['log_ret'].abs() < 0.5, np.nan)

    daily['hist_vol'] = (
        daily['log_ret'].rolling(30, min_periods=15).std() * np.sqrt(252)
    ).ffill().clip(0.05, 2.0)

    print(f"[features] Masked {len(masked_dates)} extreme return day(s) as likely "
          f"split artifacts: {masked_dates}")
    print(f"[features] hist_vol range after fix: {daily['hist_vol'].min():.4f} "
          f"to {daily['hist_vol'].max():.4f}")

    df = df.merge(daily[['[QUOTE_DATE]', 'hist_vol']], on='[QUOTE_DATE]', how='left')

    # Fill NaN volume/bid fields before filtering
    for col in ['[C_VOLUME]', '[P_VOLUME]', '[C_BID]', '[P_BID]']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Filter T <= 0
    df = df[df['T'] > 0]

    # Per-type bad-quote masks
    call_ok = (df['call_mid'] > 0) & ~((df['[C_VOLUME]'] == 0) & (df['[C_BID]'] == 0))
    put_ok  = (df['put_mid']  > 0) & ~((df['[P_VOLUME]'] == 0) & (df['[P_BID]'] == 0))

    call_df = df[call_ok].copy()
    put_df  = df[put_ok].copy()

    call_df['opttype']        = 'C'
    call_df['opttype_encoded'] = 0
    call_df['mid']            = call_df['call_mid']
    call_df['bid_ask_spread'] = call_df['[C_ASK]'] - call_df['[C_BID]']
    call_df['volume']         = call_df['[C_VOLUME]']
    call_df['iv']             = call_df['[C_IV]']

    put_df['opttype']         = 'P'
    put_df['opttype_encoded'] = 1
    put_df['mid']             = put_df['put_mid']
    put_df['bid_ask_spread']  = put_df['[P_ASK]'] - put_df['[P_BID]']
    put_df['volume']          = put_df['[P_VOLUME]']
    put_df['iv']              = put_df['[P_IV]']

    rename = {
        '[QUOTE_DATE]':      'quote_date',
        '[UNDERLYING_LAST]': 'underlying',
        '[STRIKE]':          'strike',
    }
    src_cols = [
        '[QUOTE_DATE]', '[UNDERLYING_LAST]', '[STRIKE]', 'T', 'log_moneyness',
        'hist_vol', 'opttype', 'opttype_encoded', 'bid_ask_spread', 'volume', 'iv', 'mid',
    ]

    call_df = call_df[src_cols].rename(columns=rename)
    put_df  = put_df[src_cols].rename(columns=rename)

    result = pd.concat([call_df, put_df], ignore_index=True)
    result['r'] = R

    # Post-melt liquidity filters
    print(f"[features] Rows before liquidity filters: {len(result):,}")

    n0 = len(result)
    result = result[result['T'] > 2 / 365]
    print(f"[features] DTE >= 2 filter: {n0:,} -> {len(result):,}")

    n0 = len(result)
    result = result[result['bid_ask_spread'] / (result['mid'] + 1e-6) < 0.5]
    print(f"[features] spread/mid < 0.5 filter: {n0:,} -> {len(result):,}")

    n0 = len(result)
    result = result[result['log_moneyness'].abs() < 1.0]
    print(f"[features] |log_moneyness| < 1.0 filter: {n0:,} -> {len(result):,}")

    final_cols = [
        'quote_date', 'underlying', 'strike', 'T', 'r', 'hist_vol',
        'log_moneyness', 'opttype_encoded', 'bid_ask_spread', 'volume', 'iv', 'mid',
    ]
    result = result[final_cols].sort_values('quote_date').reset_index(drop=True)
    print(f"[features] Final shape: {result.shape}")
    return result


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    df = load_features(ROOT / 'nvda_cleaned_2021_2022.csv')
    print(df.shape)
    print(df.dtypes)
    print(df.head(3))
