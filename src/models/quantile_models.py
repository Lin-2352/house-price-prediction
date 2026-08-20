"""Quantile-regression triplet used for the CQR (adaptive-width) conformal
arm. MAPIE's ConformalizedQuantileRegressor only accepts
sklearn.linear_model.QuantileRegressor, sklearn.ensemble.
GradientBoostingRegressor, sklearn.ensemble.HistGradientBoostingRegressor,
or lightgbm.LGBMRegressor as its base estimator -- XGBoost is not on that
whitelist (verified against the installed mapie==1.4.1 source). We use
HistGradientBoostingRegressor here, and also run split-conformal on this
same model's median output (see uncertainty/conformal.py), so one base
model is held constant across both interval methods and the width-vs-
coverage comparison isolates "does adaptivity help" from "does the base
model differ".
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.config import CONFIDENCE_LEVEL, RANDOM_SEED
from src.features.common import FeatureSpec, build_preprocessor

# MAPIE's required order for prefit CQR estimators: [alpha/2, 1-alpha/2, 0.5]
ALPHA = 1 - CONFIDENCE_LEVEL
QUANTILES = [ALPHA / 2, 1 - ALPHA / 2, 0.5]  # e.g. [0.05, 0.95, 0.5]


def build_quantile_pipeline(spec: FeatureSpec, quantile: float) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(spec)),
        ("reg", HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, random_state=RANDOM_SEED,
            max_iter=300, learning_rate=0.05,
        )),
    ])


def train_quantile_triplet(spec: FeatureSpec, X_train, y_train_log) -> list[Pipeline]:
    """Fits and returns [lower, upper, median] pipelines, in the exact
    order MAPIE's ConformalizedQuantileRegressor(prefit=True) requires."""
    triplet = []
    for q in QUANTILES:
        print(f"  fitting quantile-{q:.2f} HistGBR...")
        pipe = build_quantile_pipeline(spec, q)
        pipe.fit(X_train, y_train_log)
        triplet.append(pipe)
    return triplet
