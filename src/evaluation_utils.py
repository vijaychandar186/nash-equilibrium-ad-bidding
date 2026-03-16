import numpy as np
from sklearn.metrics import mean_squared_error

def quantile_loss(y_true, y_pred, quantile):
    """Calculate quantile loss for a given quantile."""
    residual = y_true - y_pred
    return np.maximum(quantile * residual, (quantile - 1) * residual).mean()

def evaluate_model(name, y_true, y_pred_log, verbose=True):
    """
    Evaluate model performance on original scale.

    Args:
        name: Model name
        y_true: True CPM values (original scale)
        y_pred_log: Predicted CPM values (log scale)
        verbose: Print results

    Returns:
        Dictionary of metrics
    """
    # Inverse transform predictions
    y_pred = np.expm1(y_pred_log)
    y_pred[y_pred < 0] = 0  # Ensure no negative predictions

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE (only for non-zero true values)
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    # Quantile Losses
    q10 = quantile_loss(y_true, y_pred, 0.1)
    q50 = quantile_loss(y_true, y_pred, 0.5)
    q90 = quantile_loss(y_true, y_pred, 0.9)

    if verbose:
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        print(f"RMSE:              {rmse:.4f}")
        print(f"MAPE:              {mape:.4f}%")
        print(f"Quantile Loss (Q10): {q10:.4f}")
        print(f"Quantile Loss (Q50): {q50:.4f}")
        print(f"Quantile Loss (Q90): {q90:.4f}")

    return {
        "Model": name,
        "RMSE": rmse,
        "MAPE": mape,
        "Q10": q10,
        "Q50": q50,
        "Q90": q90,
        "predictions": y_pred  # Store for later analysis
    }
