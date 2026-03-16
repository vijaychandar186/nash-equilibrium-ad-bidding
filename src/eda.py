import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd
import seaborn as sns

def impression_report(ad_data):
    impression_stats = PrettyTable()
    impression_stats.field_names = ["Total","Viewable","Measurable"]
    impression_stats.align[""] = "r"
    impression_stats.add_row(["100%",
                              "%.2f"%((sum(ad_data['viewable_impressions'])/sum(ad_data['total_impressions']))*100),
                              "%.2f"%((sum(ad_data['measurable_impressions'])/sum(ad_data['total_impressions']))*100)])
    print("Impression Report\n")
    print(impression_stats)

def visualizations(ad_data):
    # Visualize CPM distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Raw CPM distribution (zoomed to 99th percentile)
    axes[0, 0].hist(ad_data['CPM'], bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlim(0, ad_data['CPM'].quantile(0.99))
    axes[0, 0].set_xlabel('CPM')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('CPM Distribution (Raw, 99th percentile cutoff)')
    axes[0, 0].axvline(ad_data['CPM'].median(), color='red', linestyle='--',
                       label=f'Median: {ad_data["CPM"].median():.2f}')
    axes[0, 0].legend()

    # 2. Log1p CPM distribution
    axes[0, 1].hist(np.log1p(ad_data['CPM']), bins=100, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Log1p(CPM)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Log-Transformed CPM Distribution')

    # 3. Non-zero CPM only
    nonzero_cpm = ad_data[ad_data['CPM'] > 0]['CPM']
    axes[1, 0].hist(nonzero_cpm, bins=100, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlim(0, nonzero_cpm.quantile(0.99))
    axes[1, 0].set_xlabel('CPM (Non-Zero Only)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Non-Zero CPM Distribution (n={len(nonzero_cpm):,})')

    # 4. Box plot
    axes[1, 1].boxplot([ad_data['CPM']], vert=True, patch_artist=True)
    axes[1, 1].set_ylabel('CPM')
    axes[1, 1].set_title('CPM Box Plot (showing outliers)')
    axes[1, 1].set_xticklabels(['CPM'])

    plt.tight_layout()
    plt.show()

def cardinality_and_numeric(ad_data):
    zero_count = (ad_data['CPM'] == 0).sum()
    total_count = len(ad_data)
    zero_pct = zero_count / total_count

    print(f"\nObservation: Severe zero-inflation ({zero_pct:.1%}) and heavy right-skew detected")
    print(f"Justifies log-transform and two-stage modeling approach")

    # Analyze categorical feature cardinality
    categorical_cols = ['site_id', 'ad_type_id', 'geo_id', 'device_category_id', 'advertiser_id',
                        'os_id', 'monetization_channel_id', 'ad_unit_id']

    print("Categorical Feature Cardinality")
    print("="*60)
    cardinality_info = []
    for col in categorical_cols:
        if col in ad_data.columns:
            n_unique = ad_data[col].nunique()
            cardinality_info.append({'Feature': col, 'Unique_Values': n_unique})
            print(f"{col:30s}: {n_unique:6d} unique values")
    cardinality_df = pd.DataFrame(cardinality_info)

    # Visualize cardinality
    plt.figure(figsize=(10, 5))
    plt.barh(cardinality_df['Feature'], cardinality_df['Unique_Values'])
    plt.xlabel('Number of Unique Values')
    plt.title('Categorical Feature Cardinality (Log Scale)')
    plt.xscale('log')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

    # Analyze numeric features
    numeric_cols = ['total_impressions', 'viewable_impressions', 'measurable_impressions']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, col in enumerate(numeric_cols):
        if col in ad_data.columns:
            axes[i].hist(ad_data[col], bins=50, edgecolor='black', alpha=0.7)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency (log scale)')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_yscale('log')

    plt.tight_layout()
    plt.show()

    print("\nNumeric Feature Statistics")
    print("="*60)
    print(ad_data[numeric_cols].describe())

    # Correlation matrix for numeric features
    numeric_features = ad_data.select_dtypes(include=[np.number])
    corr_matrix = numeric_features.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix (Numeric Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # High correlations
    print("\nHighly Correlated Feature Pairs (|r| > 0.8)")
    print("="*60)
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': f"{corr_matrix.iloc[i, j]:.3f}"
                })

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr)
        print(high_corr_df.to_string(index=False))
        print(f"\nNote: total_impressions, viewable_impressions, measurable_impressions are highly correlated")
        print(f"This is expected as they measure the same concept. Retaining all features.")
    else:
        print("No pairs with |correlation| > 0.8 found.")

if __name__ == "__main__":
    import pickle
    import os

    # Load data from previous step
    data_path = os.path.join(os.path.dirname(__file__), "../data/ad_data_featured.pkl")
    print(f"Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        ad_data = pickle.load(f)

    # Run EDA functions
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)

    impression_report(ad_data)
    print("\n✓ Skipping visualizations in pipeline mode")
    # Cardinality analysis (without plots)
    categorical_cols = ['site_id', 'ad_type_id', 'geo_id', 'device_category_id', 'advertiser_id',
                        'os_id', 'monetization_channel_id', 'ad_unit_id']

    print("\nCategorical Feature Cardinality")
    print("="*60)
    for col in categorical_cols:
        if col in ad_data.columns:
            n_unique = ad_data[col].nunique()
            print(f"{col:30s}: {n_unique:6d} unique values")

    # Numeric features
    numeric_cols = ['total_impressions', 'viewable_impressions', 'measurable_impressions']
    print("\nNumeric Feature Statistics")
    print("="*60)
    print(ad_data[numeric_cols].describe())

    print("\n✓ EDA complete")
