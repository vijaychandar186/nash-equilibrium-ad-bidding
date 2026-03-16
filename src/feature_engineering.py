import numpy as np
import pandas as pd
import pickle
import os

def safe_division(n, d):
    """Safely divide, returning 0 if denominator is 0"""
    return n / d if d else 0

def create_features(ad_data):
    # Calculate CPM
    ad_data['CPM'] = ad_data.apply(
        lambda x: safe_division((x['total_revenue'] * 100), x['measurable_impressions']) * 1000,
        axis=1
    )

    # Create derived feature: Viewability ratio (indicates ad quality)
    ad_data['View_Measurable_Ratio'] = ad_data.apply(
        lambda x: safe_division(x['viewable_impressions'], x['measurable_impressions']),
        axis=1
    )

    # Convert date to timestamp for train/test split
    ad_data['date'] = ad_data.date.apply(lambda l: pd.Timestamp(l).value)

    print("✓ CPM and derived features created")
    return ad_data

def zero_inflation_analysis(ad_data):
    # Analyze zero inflation
    zero_cpm_count = (ad_data['CPM'] == 0).sum()
    zero_cpm_pct = (zero_cpm_count / len(ad_data)) * 100

    print("=" * 60)
    print("ZERO-INFLATION ANALYSIS")
    print("=" * 60)
    print(f"Records with CPM = 0: {zero_cpm_count:,} ({zero_cpm_pct:.2f}%)")
    print(f"Records with CPM > 0: {len(ad_data) - zero_cpm_count:,} ({100-zero_cpm_pct:.2f}%)")
    print(f"\nCPM Statistics (All Data):")
    print(ad_data['CPM'].describe())
    print(f"\nCPM Statistics (Non-Zero Only):")
    print(ad_data[ad_data['CPM'] > 0]['CPM'].describe())

if __name__ == "__main__":
    # Load data from previous step
    data_path = os.path.join(os.path.dirname(__file__), "../data/ad_data_loaded.pkl")
    print(f"Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        ad_data = pickle.load(f)

    # Create features
    ad_data = create_features(ad_data)

    # Run zero inflation analysis
    zero_inflation_analysis(ad_data)

    # Save processed data
    output_path = os.path.join(os.path.dirname(__file__), "../data/ad_data_featured.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(ad_data, f)
    print(f"\n✓ Saved featured data to: {output_path}")