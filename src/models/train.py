"""Train and tune the three point-prediction models (linear baseline,
Random Forest, XGBoost) on the training split only.

Each model is a full sklearn Pipeline (shared preprocessor + regressor),
so the fitted object can be handed directly to MAPIE's `prefit=True`
conformal wrappers and to SHAP without re-deriving preprocessing logic
anywhere else. Hyperparameters are tuned by K-fold cross-validation
*within* the training split -- calibration and test rows are never seen
during this step.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import RANDOM_SEED
from src.features.common import FeatureSpec, build_preprocessor


def _cv():
    return KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def build_linear_pipeline(spec: FeatureSpec) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(spec)),
        ("reg", Ridge(random_state=RANDOM_SEED)),
    ])


def build_rf_pipeline(spec: FeatureSpec) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(spec)),
        ("reg", RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)),
    ])


def build_xgb_pipeline(spec: FeatureSpec) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(spec)),
        ("reg", XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist")),
    ])


LINEAR_GRID = {"reg__alpha": np.logspace(-2, 3, 20)}

RF_GRID = {
    "reg__n_estimators": [200, 400, 600, 800],
    "reg__max_depth": [None, 8, 12, 16, 24],
    "reg__min_samples_leaf": [1, 2, 4, 8],
    "reg__max_features": ["sqrt", 0.5, 0.7, 1.0],
}

XGB_GRID = {
    "reg__n_estimators": [200, 400, 600, 900],
    "reg__max_depth": [3, 4, 5, 6, 8],
    "reg__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "reg__subsample": [0.6, 0.8, 1.0],
    "reg__colsample_bytree": [0.6, 0.8, 1.0],
    "reg__reg_lambda": [0.5, 1.0, 2.0, 5.0],
}


def _tune(pipeline: Pipeline, grid: dict, X, y, n_iter: int) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        pipeline, grid, n_iter=n_iter, cv=_cv(),
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED, n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    return search


def train_all(spec: FeatureSpec, X_train, y_train_log) -> dict:
    """Returns {"linear": fitted_pipeline, "random_forest": ..., "xgboost": ...,
    "cv_rmse_log": {name: score}} -- all three are kept (not just the
    winner) since the synopsis requires a baseline-vs-tuned comparison."""
    results = {}
    cv_scores = {}

    print("Tuning linear (Ridge) baseline...")
    search = _tune(build_linear_pipeline(spec), LINEAR_GRID, X_train, y_train_log, n_iter=20)
    results["linear"] = search.best_estimator_
    cv_scores["linear"] = -search.best_score_
    print(f"  best alpha={search.best_params_}, CV RMSE(log)={-search.best_score_:.4f}")

    print("Tuning random forest...")
    search = _tune(build_rf_pipeline(spec), RF_GRID, X_train, y_train_log, n_iter=20)
    results["random_forest"] = search.best_estimator_
    cv_scores["random_forest"] = -search.best_score_
    print(f"  best params={search.best_params_}, CV RMSE(log)={-search.best_score_:.4f}")

    print("Tuning XGBoost...")
    search = _tune(build_xgb_pipeline(spec), XGB_GRID, X_train, y_train_log, n_iter=25)
    results["xgboost"] = search.best_estimator_
    cv_scores["xgboost"] = -search.best_score_
    print(f"  best params={search.best_params_}, CV RMSE(log)={-search.best_score_:.4f}")

    results["cv_rmse_log"] = cv_scores
    best_name = min(cv_scores, key=cv_scores.get)
    results["best_name"] = best_name
    print(f"Best model by CV RMSE(log): {best_name}")
    return results
