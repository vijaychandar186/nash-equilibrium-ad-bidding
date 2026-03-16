import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def diebold_mariano_test(errors1, errors2, h=1):
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Args:
        errors1: Forecast errors from model 1 (y_true - y_pred_1)
        errors2: Forecast errors from model 2 (y_true - y_pred_2)
        h: Forecast horizon (default 1 for one-step ahead)

    Returns:
        DM statistic and p-value
    """
    # Squared errors (MSE loss)
    d = errors1**2 - errors2**2

    # Mean difference
    mean_d = np.mean(d)

    # Variance of difference
    var_d = np.var(d, ddof=1)

    # DM statistic
    n = len(d)
    dm_stat = mean_d / np.sqrt(var_d / n)

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value

def summarize_results(results):
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.drop('predictions', axis=1)

    results_df['balance_score'] = (
        (results_df['RMSE'] / results_df['RMSE'].max()) +
        (results_df['MAPE'] / results_df['MAPE'].max()) +
        (results_df['Q50'] / results_df['Q50'].max())
    )

    results_df_sorted = results_df.sort_values(by=['balance_score', 'RMSE'])
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(results_df_sorted.drop('balance_score', axis=1).to_string(index=False))
    print("="*80)

    print("\nBEST PERFORMERS:")
    print(f"Best Balanced: {results_df_sorted.iloc[0]['Model']}")
    print(f"Best RMSE:  {results_df.loc[results_df['RMSE'].idxmin()]['Model']} ({results_df['RMSE'].min():.4f})")
    print(f"Best MAPE:  {results_df.loc[results_df['MAPE'].idxmin()]['Model']} ({results_df['MAPE'].min():.4f}%)")
    print(f"Best Q50:   {results_df.loc[results_df['Q50'].idxmin()]['Model']} ({results_df['Q50'].min():.4f})")
    return results_df, results_df_sorted

def confusion_and_classification(prob_test_nonzero, y_test_log):
    print("="*80)
    print("CONFUSION MATRIX: Two-Stage LightGBM Classifier (Best Model)")
    print("="*80)

    y_test_binary = (y_test_log > 0).astype(int)
    y_pred_binary = (prob_test_nonzero > 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test_binary, y_pred_binary)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test_binary, y_pred_binary,
                               target_names=['Zero CPM', 'Non-Zero CPM']))

    # Visualize Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Zero', 'Predicted Non-Zero'],
                yticklabels=['Actual Zero', 'Actual Non-Zero'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix: Zero vs Non-Zero CPM Classification',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"\nKey Metrics:")
    print(f"True Negatives (correctly predicted zero):  {tn:,}")
    print(f"False Positives (predicted non-zero, was zero): {fp:,}")
    print(f"False Negatives (predicted zero, was non-zero): {fn:,}")
    print(f"True Positives (correctly predicted non-zero): {tp:,}")
    print(f"\nAccuracy: {(tn + tp) / (tn + fp + fn + tp):.4f}")
    print(f"Precision (Non-Zero): {tp / (tp + fp):.4f}")
    print(f"Recall (Non-Zero): {tp / (tp + fn):.4f}")

def pairwise_dm_tests(results_full, results_df_sorted, y_test_true):
    # Select top 4 models by balance_score
    top_models = results_df_sorted.head(4)['Model'].tolist()
    print(f"\nComparing top 4 models:")
    for i, model in enumerate(top_models, 1):
        print(f"{i}. {model}")

    comparison_results = []
    for i, model1 in enumerate(top_models):
        for j, model2 in enumerate(top_models):
            if i < j:
                pred1 = results_full[results_full['Model'] == model1]['predictions'].values[0]
                pred2 = results_full[results_full['Model'] == model2]['predictions'].values[0]

                errors1 = y_test_true.values - pred1
                errors2 = y_test_true.values - pred2

                dm_stat, p_value = diebold_mariano_test(errors1, errors2)

                significant = "Yes" if p_value < 0.05 else "No"
                better = model2 if dm_stat > 0 else model1

                comparison_results.append({
                    'Model 1': model1,
                    'Model 2': model2,
                    'DM Stat': f"{dm_stat:.4f}",
                    'p-value': f"{p_value:.4f}",
                    'Significant': significant,
                    'Better Model': better if significant == "Yes" else "No difference"
                })

    comparison_df = pd.DataFrame(comparison_results)
    print(comparison_df.to_string(index=False))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    sig_comparisons = comparison_df[comparison_df['Significant'] == 'Yes']
    if len(sig_comparisons) > 0:
        print(f"\nFound {len(sig_comparisons)} statistically significant differences (p < 0.05):")
        for idx, row in sig_comparisons.iterrows():
            print(f"  • {row['Better Model']} significantly outperforms the alternative")
            print(f"    (DM = {row['DM Stat']}, p = {row['p-value']})")
    else:
        print("\nNo statistically significant differences found among top models.")
        print("All models perform similarly from a statistical perspective.")

if __name__ == "__main__":
    import pickle
    import os

    # Load results from previous step
    results_path = os.path.join(os.path.dirname(__file__), "../data/model_results.pkl")
    print(f"Loading model results from: {results_path}")
    with open(results_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    prob_test_nonzero = data['prob_test_nonzero']
    y_test_log = data['y_test_log']
    y_test_true = data['y_test_true']

    print("\n" + "="*80)
    print("RESULTS AND STATISTICAL ANALYSIS")
    print("="*80)

    # Summarize results
    results_df, results_df_sorted = summarize_results(results)

    # Confusion matrix for two-stage model
    print("\n")
    print("✓ Skipping confusion matrix visualization in headless mode")
    # confusion_and_classification(prob_test_nonzero, y_test_log)

    # Pairwise DM tests
    print("\n" + "="*80)
    print("DIEBOLD-MARIANO PAIRWISE TESTS")
    print("="*80)
    pairwise_dm_tests(pd.DataFrame(results), results_df_sorted, y_test_true)

    print("\n✓ Results and statistics complete")
