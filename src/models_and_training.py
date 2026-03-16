import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from evaluation_utils import evaluate_model, quantile_loss
    from train_test_split_and_imputation import target_encode
except ImportError:
    from src.evaluation_utils import evaluate_model, quantile_loss
    from src.train_test_split_and_imputation import target_encode 

# Custom asymmetric loss from your notebook
def custom_asymmetric_loss(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * 2.5 * residual, -2 * residual)
    hess = np.where(residual < 0, 2 * 5.0, 2.0)
    return grad, hess

def train_ridge(X_train_ridge, X_test_ridge, y_train_log):
    scaler = StandardScaler()
    X_train_ridge_scaled = scaler.fit_transform(X_train_ridge)
    X_test_ridge_scaled = scaler.transform(X_test_ridge)

    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    ridge_search = RandomizedSearchCV(ridge, ridge_params, n_iter=4, cv=3, scoring='neg_root_mean_squared_error', random_state=42, verbose=0)
    ridge_search.fit(X_train_ridge_scaled, y_train_log)

    print(f"✓ Best params: {ridge_search.best_params_}")
    best_ridge = ridge_search.best_estimator_
    pred_ridge_log = best_ridge.predict(X_test_ridge_scaled)
    return pred_ridge_log, ridge_search.best_params_

# H2O AutoML (as in notebook)
def train_h2o_automl(X_train, y_train_log, X_test, target='CPM'):
    import h2o
    from h2o.automl import H2OAutoML
    print("Initializing H2O AutoML...")
    h2o.init()

    train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train_log.rename(target)], axis=1))
    test_h2o = h2o.H2OFrame(X_test)

    aml = H2OAutoML(max_models=10, seed=1, stopping_metric='MSE', nfolds=0)
    print("\nTraining H2O AutoML (this may take a few minutes)...")
    aml.train(x=list(X_train.columns), y=target, training_frame=train_h2o)

    print("\n H2O AutoML Leaderboard (Top 10 Models):")
    print(aml.leaderboard.head(10))

    pred_h2o_log = aml.predict(test_h2o)
    pred_h2o_log_np = h2o.as_list(pred_h2o_log).values.flatten()
    return pred_h2o_log_np

def train_random_forest(X_train_ridge, y_train_log, X_test_ridge):
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_params,
        n_iter=10,
        cv=3,
        scoring='neg_root_mean_squared_error',
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    rf_search.fit(X_train_ridge, y_train_log)
    print(f"✓ Best params: {rf_search.best_params_}")
    best_rf = rf_search.best_estimator_
    pred_rf_log = best_rf.predict(X_test_ridge)
    return pred_rf_log

def train_xgboost(X_train, y_train_log, X_test):
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 9],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    xgb_search = RandomizedSearchCV(
        estimator=xgbr,
        param_distributions=xgb_params,
        n_iter=10,
        cv=3,
        scoring='neg_root_mean_squared_error',
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(X_train.values, y_train_log.values)
    print(f"✓ Best params: {xgb_search.best_params_}")
    best_xgb = xgb_search.best_estimator_
    pred_xgb_log = best_xgb.predict(X_test.values)
    return pred_xgb_log

def train_catboost(X_train, y_train_log, X_test, cat_cols):
    cat_model = CatBoostRegressor(
        iterations=500,
        cat_features=cat_cols,
        verbose=0,
        random_state=42
    )
    cat_model.fit(X_train, y_train_log)
    print(f"✓ Training complete (500 iterations)")
    pred_cat_log = cat_model.predict(X_test)
    return pred_cat_log

def train_lightgbm(X_train_lgb, y_train_log, X_test_lgb):
    lgb_params = {
        'n_estimators': [1000, 2000, 3000],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'min_child_samples': [20, 50, 100],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0, 0.1, 1.0]
    }
    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
    lgb_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=lgb_params,
        n_iter=10,
        cv=3,
        scoring='neg_root_mean_squared_error',
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    lgb_search.fit(X_train_lgb, y_train_log)
    print(f"✓ Best params: {lgb_search.best_params_}")
    best_lgb = lgb_search.best_estimator_
    pred_lgb_log = best_lgb.predict(X_test_lgb)
    return pred_lgb_log, lgb_search.best_params_

def train_lightgbm_asym(X_train_lgb, y_train_log, X_test_lgb):
    lgb_asym = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        objective=custom_asymmetric_loss,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    lgb_asym.fit(
        X_train_lgb, y_train_log,
        eval_set=[(X_test_lgb, y_train_log)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False)
        ]
    )
    print(f"✓ Training complete with early stopping")
    pred_lgb_asym_log = lgb_asym.predict(X_test_lgb)
    return pred_lgb_asym_log

def train_two_stage_pipeline(X_train, X_test, X_train_lgb, X_test_lgb, y_train_log, y_test_log, y_test_true, lgb_search):
    # Stage 1: Classifier (Zero vs Non-Zero) using XGBoost
    y_train_binary = (y_train_log > 0).astype(int)

    clf_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1]
    }

    clf = XGBClassifier(objective='binary:logistic', random_state=42, n_jobs=-1, eval_metric='logloss')
    clf_search = RandomizedSearchCV(clf, clf_params, n_iter=4, cv=3, scoring='roc_auc', verbose=0, random_state=42)
    clf_search.fit(X_train, y_train_binary)
    print(f"✓ Best Classifier Params: {clf_search.best_params_}")
    best_clf = clf_search.best_estimator_
    prob_test_nonzero = best_clf.predict_proba(X_test)[:, 1]

    # Stage 2: Regressor
    print("Training Stage 2 Regressor (using best LightGBM params)...")
    mask_train_nonzero = y_train_log > 0
    X_train_nonzero = X_train_lgb[mask_train_nonzero]
    y_train_log_nonzero = y_train_log[mask_train_nonzero]

    stage2_params = lgb_search.best_params_
    lgb_stage2 = lgb.LGBMRegressor(**stage2_params, random_state=42, n_jobs=-1, verbosity=-1)
    lgb_stage2.fit(X_train_nonzero, y_train_log_nonzero)
    pred_stage2_log = lgb_stage2.predict(X_test_lgb)

    # Combine
    pred_stage2_linear = np.expm1(pred_stage2_log)
    pred_twostage_linear = prob_test_nonzero * pred_stage2_linear
    pred_twostage_log = np.log1p(pred_twostage_linear)
    print("✓ Two-stage pipeline complete")
    return pred_twostage_log, prob_test_nonzero

if __name__ == "__main__":
    import pickle
    import os

    # Load data from previous step
    data_path = os.path.join(os.path.dirname(__file__), "../data/train_test_data.pkl")
    print(f"Loading train/test data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    train_df = data['train_df']
    test_df = data['test_df']
    X_train = data['X_train']
    X_test = data['X_test']
    y_train_log = data['y_train_log']
    y_test_log = data['y_test_log']
    y_test_true = data['y_test_true']
    features = data['features']
    cat_cols = data['cat_cols']
    target = data['target']

    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)

    # Prepare data variants for different models
    # Target-encoded data for Ridge and Random Forest
    X_train_ridge = X_train.copy()
    X_test_ridge = X_test.copy()
    for col in cat_cols:
        if col in X_train.columns:
            X_train_ridge[col], X_test_ridge[col] = target_encode(train_df, test_df, col, target)

    # LightGBM requires target encoding too (it doesn't support categorical natively in all contexts)
    X_train_lgb = X_train.copy()
    X_test_lgb = X_test.copy()
    for col in cat_cols:
        if col in X_train.columns:
            X_train_lgb[col], X_test_lgb[col] = target_encode(train_df, test_df, col, target)

    # Store results
    results = []

    # Train Ridge Regression
    print("\n[1/7] Ridge Regression")
    pred_ridge_log, ridge_params = train_ridge(X_train_ridge, X_test_ridge, y_train_log)
    results.append(evaluate_model("Ridge Regression", y_test_true, pred_ridge_log, verbose=False))

    # Train Random Forest
    print("\n[2/7] Random Forest")
    pred_rf_log = train_random_forest(X_train_ridge, y_train_log, X_test_ridge)
    results.append(evaluate_model("Random Forest", y_test_true, pred_rf_log, verbose=False))

    # Train XGBoost
    print("\n[3/7] XGBoost")
    pred_xgb_log = train_xgboost(X_train, y_train_log, X_test)
    results.append(evaluate_model("XGBoost", y_test_true, pred_xgb_log, verbose=False))

    # Train CatBoost
    print("\n[4/7] CatBoost")
    pred_cat_log = train_catboost(X_train, y_train_log, X_test, cat_cols)
    results.append(evaluate_model("CatBoost", y_test_true, pred_cat_log, verbose=False))

    # Train LightGBM
    print("\n[5/7] LightGBM")
    pred_lgb_log, lgb_params = train_lightgbm(X_train_lgb, y_train_log, X_test_lgb)
    results.append(evaluate_model("LightGBM", y_test_true, pred_lgb_log, verbose=False))

    # Train LightGBM with Asymmetric Loss
    print("\n[6/7] LightGBM (Asymmetric Loss)")
    pred_lgb_asym_log = train_lightgbm_asym(X_train_lgb, y_train_log, X_test_lgb)
    results.append(evaluate_model("LightGBM (Asymmetric)", y_test_true, pred_lgb_asym_log, verbose=False))

    # Train Two-Stage Pipeline
    print("\n[7/7] Two-Stage LightGBM")
    # Need to create a mock lgb_search object with best_params_
    class MockSearch:
        def __init__(self, params):
            self.best_params_ = params
    lgb_search = MockSearch(lgb_params)
    pred_twostage_log, prob_test_nonzero = train_two_stage_pipeline(
        X_train, X_test, X_train_lgb, X_test_lgb,
        y_train_log, y_test_log, y_test_true, lgb_search
    )
    results.append(evaluate_model("Two-Stage LightGBM", y_test_true, pred_twostage_log, verbose=False))

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "../data/model_results.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'prob_test_nonzero': prob_test_nonzero,
            'y_test_log': y_test_log,
            'y_test_true': y_test_true
        }, f)
    print(f"\n✓ Saved model results to: {output_path}")
    print("\n✓ All models trained successfully")