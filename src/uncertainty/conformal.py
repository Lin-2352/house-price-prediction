"""Conformal-prediction wrappers around MAPIE v1.4.1's current API
(SplitConformalRegressor / ConformalizedQuantileRegressor -- the old
MapieRegressor class is deprecated and not used here).

All fitting/calibration happens in log-price space; intervals are
exponentiated back to currency only for display (valid because empirical
coverage is invariant under a monotone transform like log/expm1).
"""
from __future__ import annotations

import numpy as np
from mapie.regression import ConformalizedQuantileRegressor, SplitConformalRegressor

from src.config import CONFIDENCE_LEVEL
from src.features.common import inverse_log1p


def fit_split_conformal(fitted_estimator, X_cal, y_cal_log) -> SplitConformalRegressor:
    """`fitted_estimator` must already be fit on the training split.
    Calibrates conformity scores on the (unseen-by-the-model) calibration
    split -- this is what makes the resulting interval's coverage
    guarantee valid."""
    scr = SplitConformalRegressor(
        estimator=fitted_estimator, confidence_level=CONFIDENCE_LEVEL, prefit=True,
    )
    scr.conformalize(X_cal, y_cal_log)
    return scr


def fit_cqr(quantile_triplet, X_cal, y_cal_log) -> ConformalizedQuantileRegressor:
    """`quantile_triplet` must be [lower, upper, median] pipelines already
    fit on the training split (see models/quantile_models.py), in exactly
    that order -- MAPIE requires it and getting it backwards silently
    inverts the interval."""
    cqr = ConformalizedQuantileRegressor(
        estimator=quantile_triplet, confidence_level=CONFIDENCE_LEVEL, prefit=True,
    )
    cqr.conformalize(X_cal, y_cal_log)
    return cqr


def predict_interval_log(conformal_obj, X):
    """Returns (point_log, lower_log, upper_log), each shape (n,)."""
    point, interval = conformal_obj.predict_interval(X)
    lower = interval[:, 0, 0]
    upper = interval[:, 1, 0]
    return np.asarray(point), np.asarray(lower), np.asarray(upper)


def predict_interval_currency(conformal_obj, X):
    """Same as predict_interval_log but exponentiated back to currency."""
    point, lower, upper = predict_interval_log(conformal_obj, X)
    return inverse_log1p(point), inverse_log1p(lower), inverse_log1p(upper)
