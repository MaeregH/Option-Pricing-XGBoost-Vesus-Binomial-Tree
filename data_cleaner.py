import pandas as pd
import os
from datetime import datetime

# Features to keep 
FEATURES_TO_KEEP = [
    '[QUOTE_DATE]',
    '[UNDERLYING_LAST]',
    '[EXPIRE_DATE]',
    '[DTE]',
    '[STRIKE]',
    '[C_BID]',
    '[C_ASK]',
    '[C_VOLUME]',
    '[P_BID]',
    '[P_ASK]',
    '[P_VOLUME]',
    '[C_IV]',
    '[P_IV]',
    '[C_LAST]',
    '[P_LAST]',
    '[STRIKE_DISTANCE]',
    '[STRIKE_DISTANCE_PCT]'
]

def clean_nvda_dataset(input_file, output_file='nvda_cleaned_2021_2022.csv'):
    #filter to years 2021-2022

    df = pd.read_csv(input_file, low_memory=False)
    df.columns = df.columns.str.strip()

    print(f"Original dataset shape: {df.shape}")
    print(f"Normalized columns: {list(df.columns)}")

    # Ensure date column exists
    if '[QUOTE_DATE]' not in df.columns:
        raise KeyError(f"'[QUOTE_DATE]' not found in columns: {df.columns.tolist()}")

    # Convert QUOTE_DATE to datetime
    print("\nConverting date columns...")
    df['[QUOTE_DATE]'] = pd.to_datetime(df['[QUOTE_DATE]'])

    # Filter for 2021-2022 date range
    print("Filtering date range (2021-2022)...")
    df_filtered = df[
        (df['[QUOTE_DATE]'] >= '2021-01-01') &
        (df['[QUOTE_DATE]'] <= '2022-12-31')
    ].copy()

    print(f"After date filter shape: {df_filtered.shape}")

    # Check which features exist in the dataset
    available_features = [col for col in FEATURES_TO_KEEP if col in df_filtered.columns]
    missing_features = [col for col in FEATURES_TO_KEEP if col not in df_filtered.columns]

    if missing_features:
        print(f"\nWarning: The following features are not in the dataset: {missing_features}")

    print(f"\nSelecting {len(available_features)} features...")
    df_cleaned = df_filtered[available_features].copy()

    # Display basic statistics
    print("\n" + "=" * 60)
    print("CLEANED DATASET SUMMARY")
    print("=" * 60)
    print(f"Final shape: {df_cleaned.shape}")
    print(f"Date range: {df_cleaned['[QUOTE_DATE]'].min()} to {df_cleaned['[QUOTE_DATE]'].max()}")
    print(f"Number of unique quote dates: {df_cleaned['[QUOTE_DATE]'].nunique()}")
    print(f"\nMissing values per column:")
    print(df_cleaned.isnull().sum())

    # Save to CSV
    print(f"\nSaving cleaned dataset to: {output_file}")
    df_cleaned.to_csv(output_file, index=False)
    print("Done!")

    return df_cleaned


def display_sample_data(df, n=5):
    """Display sample rows from the cleaned dataset."""
    print("\n" + "=" * 60)
    print(f"SAMPLE DATA (First {n} rows)")
    print("=" * 60)
    print(df.head(n))
    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes)


if __name__ == "__main__":
    INPUT_FILE = "nvda_2020_2022.csv"  
    OUTPUT_FILE = "nvda_cleaned_2021_2022.csv"

    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: File '{INPUT_FILE}' not found!")
        print("Please download the dataset and update the INPUT_FILE variable.")
        print("\nAlternatively, you can run the function directly:")
        print("  df = clean_nvda_dataset('your_file.csv', 'output_file.csv')")
    else:
        df_cleaned = clean_nvda_dataset(INPUT_FILE, OUTPUT_FILE)
        display_sample_data(df_cleaned)

        print("\n" + "=" * 60)
        print("ADDITIONAL INFORMATION")
        print("=" * 60)
        print(f"Cleaned file saved as: {OUTPUT_FILE}")
        print(f"You can now load this file with: pd.read_csv('{OUTPUT_FILE}')")
