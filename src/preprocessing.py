import pandas as pd
import pickle
import os

def remove_outliers(ad_data):
    # Compute 95th percentile
    cpm_95 = ad_data.CPM.quantile(.95)
    print(f"95th Percentile CPM: {cpm_95:.2f}")

    # Remove outliers
    before_count = len(ad_data)
    ad_data = ad_data[(ad_data.CPM >= 0) & (ad_data.CPM < cpm_95)]
    ad_data.reset_index(inplace=True, drop=True)
    after_count = len(ad_data)

    print(f"Records before outlier removal: {before_count:,}")
    print(f"Records after outlier removal: {after_count:,}")
    print(f"Records removed: {before_count - after_count:,} ({((before_count - after_count)/before_count)*100:.2f}%)")
    return ad_data

def drop_columns(ad_data):
    # Drop specified columns
    cols_to_drop = ['integration_type_id', 'revenue_share_percent', 'order_id', 'line_item_type_id', 'total_revenue']
    existing_cols_to_drop = [c for c in cols_to_drop if c in ad_data.columns]
    ad_data.drop(columns=existing_cols_to_drop, inplace=True)

    print(f"✓ Remaining columns: {list(ad_data.columns)}")
    return ad_data

if __name__ == "__main__":
    # Load data from previous step
    data_path = os.path.join(os.path.dirname(__file__), "../data/ad_data_featured.pkl")
    print(f"Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        ad_data = pickle.load(f)

    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)

    # Remove outliers
    ad_data = remove_outliers(ad_data)

    # Drop unnecessary columns
    ad_data = drop_columns(ad_data)

    # Save preprocessed data
    output_path = os.path.join(os.path.dirname(__file__), "../data/ad_data_preprocessed.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(ad_data, f)
    print(f"\n✓ Saved preprocessed data to: {output_path}")