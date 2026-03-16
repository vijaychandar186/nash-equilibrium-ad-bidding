import pandas as pd
import pickle
import os

def load_dataset():
    # Local dataset path
    path = os.path.join(os.path.dirname(__file__), "../data/Dataset.csv")
    path = os.path.abspath(path)
    print(f"Loading dataset from: {path}")

    ad_data = pd.read_csv(path)

    print(f"\nFirst few rows:")
    print(ad_data.head())

    # Data Overview
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total Records: {len(ad_data):,}")
    print(f"Features: {ad_data.shape[1]}")
    print(f"\nMissing Values per Column:")
    print(ad_data.isnull().sum())
    print(ad_data.dtypes)
    return ad_data, path

if __name__ == "__main__":
    ad_data, path = load_dataset()

    # Save loaded data for next step
    output_path = os.path.join(os.path.dirname(__file__), "../data/ad_data_loaded.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(ad_data, f)
    print(f"\n✓ Saved loaded data to: {output_path}")
