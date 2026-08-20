"""Two baselines the conformal ranges are compared against (required by the
synopsis): a model with no range at all, and a naive fixed-width band.

The naive baseline uses one constant *currency* half-width (the 90th
percentile of absolute calibration residuals, in currency units) applied
identically to every test-set prediction, regardless of the house's price
or how unusual it is. This is deliberately different in kind from the
conformal arms: split-conformal on a log-target model is a fixed
*additive log-space* width, which is already a fixed *multiplicative*
percentage band in currency space and adapts price-proportionally; the
naive baseline here is a flat currency amount, so it's expected to be far
too narrow for expensive houses and wastefully wide for cheap ones -- that
gap is the evidence the report is meant to show.
"""
from __future__ import annotations

import numpy as np

from src.features.common import inverse_log1p


def no_range_point_metrics(y_true_currency, point_pred_currency) -> dict:
    err = point_pred_currency - y_true_currency
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err / y_true_currency)) * 100),
    }


def calibrate_fixed_width(y_cal_currency, point_pred_cal_currency, confidence_level: float) -> float:
    """Returns the single constant currency half-width whose band would
    have covered `confidence_level` of the calibration residuals."""
    abs_resid = np.abs(y_cal_currency - point_pred_cal_currency)
    return float(np.quantile(abs_resid, confidence_level))


def predict_fixed_width(point_pred_currency, half_width: float):
    lower = point_pred_currency - half_width
    upper = point_pred_currency + half_width
    return lower, upper


def coverage_and_width(y_true_currency, lower, upper) -> dict:
    covered = (y_true_currency >= lower) & (y_true_currency <= upper)
    width = upper - lower
    return {
        "coverage": float(np.mean(covered)),
        "avg_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
    }
