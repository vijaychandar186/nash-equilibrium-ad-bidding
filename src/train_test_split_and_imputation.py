import pandas as pd
import numpy as np

# Define Features and Target
cat_cols = ['measurable_impressions', 'site_id', 'ad_type_id', 'geo_id',
            'device_category_id', 'advertiser_id', 'os_id',
            'monetization_channel_id', 'ad_unit_id']
features = cat_cols + ['total_impressions', 'viewable_impressions']
target = 'CPM'

def split_and_impute(ad_data):
    # Split Data (June 22, 2019)
    split_date = pd.Timestamp('06-22-2019').value

    train_df = ad_data.loc[ad_data.date < split_date].copy()
    test_df = ad_data.loc[ad_data.date >= split_date].copy()

    # Intelligent Missing Data Handling
    cols_with_missing = [col for col in train_df.columns if train_df[col].isnull().any()]

    for col in cols_with_missing:
        train_df[f'{col}_is_missing'] = train_df[col].isnull().astype(int)
        test_df[f'{col}_is_missing'] = test_df[col].isnull().astype(int)

    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)

    print("=" * 60)
    print("TRAIN-TEST SPLIT SUMMARY")
    print("=" * 60)
    print(f"Training Set: {train_df.shape[0]:,} records ({(train_df.shape[0]/len(ad_data))*100:.1f}%)")
    print(f"Test Set: {test_df.shape[0]:,} records ({(test_df.shape[0]/len(ad_data))*100:.1f}%)")
    print(f"\nZero CPM in Training: {(train_df[target] == 0).sum():,} ({((train_df[target] == 0).sum()/len(train_df))*100:.1f}%)")
    print(f"Zero CPM in Test: {(test_df[target] == 0).sum():,} ({((test_df[target] == 0).sum()/len(test_df))*100:.1f}%)")
    return train_df, test_df, features, cat_cols, target

def log_transform_target(train_df, test_df, target='CPM'):
    y_train_log = np.log1p(train_df[target])
    y_test_log = np.log1p(test_df[target])
    y_test_true = test_df[target]
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"\nTarget Distribution (Log-Transformed):")
    print(y_train_log.describe())
    return X_train, X_test, y_train_log, y_test_log, y_test_true

# Target encoding helper used later
def target_encode(train_df, test_df, col, target_col):
    global_mean = train_df[target_col].mean()
    agg = train_df.groupby(col)[target_col].mean()
    train_encoded = train_df[col].map(agg)
    test_encoded = test_df[col].map(agg)
    train_encoded.fillna(global_mean, inplace=True)
    test_encoded.fillna(global_mean, inplace=True)
    return train_encoded, test_encoded

if __name__ == "__main__":
    import pickle
    import os

    # Load data from previous step
    data_path = os.path.join(os.path.dirname(__file__), "../data/ad_data_preprocessed.pkl")
    print(f"Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        ad_data = pickle.load(f)

    # Split and impute
    train_df, test_df, features, cat_cols, target = split_and_impute(ad_data)

    # Log transform target
    X_train, X_test, y_train_log, y_test_log, y_test_true = log_transform_target(train_df, test_df, target)

    # Save all outputs
    output_path = os.path.join(os.path.dirname(__file__), "../data/train_test_data.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'train_df': train_df,
            'test_df': test_df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train_log': y_train_log,
            'y_test_log': y_test_log,
            'y_test_true': y_test_true,
            'features': features,
            'cat_cols': cat_cols,
            'target': target
        }, f)
    print(f"\n✓ Saved train/test data to: {output_path}")