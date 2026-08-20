"""Final, test-split-touched-once evaluation.

Produces: point-accuracy metrics (log-RMSE + currency MAE/RMSE/MAPE),
coverage & average width for both conformal arms vs both baselines,
breakdowns by price bracket and by segment (neighborhood/city, only where
the segment has enough test rows), and a validation that the confidence
flag actually catches worse predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.config import MIN_SEGMENT_N, MarketConfig
from src.uncertainty.baselines import coverage_and_width, no_range_point_metrics


def price_bracket_labels(y_currency: pd.Series) -> pd.Series:
    return pd.qcut(y_currency, q=4, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4 (most expensive)"])


def point_accuracy(y_true_log, y_pred_log, y_true_currency, y_pred_currency) -> dict:
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    currency = no_range_point_metrics(y_true_currency, y_pred_currency)
    return {"rmse_log": rmse_log, **currency}


def coverage_width_table(y_true_currency, arms: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """`arms`: {arm_name: (lower_currency, upper_currency)}"""
    rows = []
    for name, (lower, upper) in arms.items():
        stats = coverage_and_width(y_true_currency, lower, upper)
        rows.append({"arm": name, **stats})
    return pd.DataFrame(rows)


def segmented_coverage_width(
    y_true_currency: pd.Series, lower, upper, segment: pd.Series, min_n: int = MIN_SEGMENT_N
) -> pd.DataFrame:
    df = pd.DataFrame({
        "y": np.asarray(y_true_currency), "lower": np.asarray(lower), "upper": np.asarray(upper),
        "segment": np.asarray(segment),
    })
    rows = []
    for seg, g in df.groupby("segment"):
        n = len(g)
        if n < min_n:
            rows.append({"segment": seg, "n": n, "coverage": None, "avg_width": None,
                         "note": "insufficient data for a reliable estimate"})
            continue
        stats = coverage_and_width(g["y"], g["lower"], g["upper"])
        rows.append({"segment": seg, "n": n, **stats, "note": ""})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def bracket_accuracy(y_true_currency: pd.Series, y_pred_currency: pd.Series) -> pd.DataFrame:
    brackets = price_bracket_labels(y_true_currency)
    df = pd.DataFrame({"y": y_true_currency, "pred": y_pred_currency, "bracket": brackets})
    rows = []
    for b, g in df.groupby("bracket", observed=True):
        m = no_range_point_metrics(g["y"], g["pred"])
        rows.append({"bracket": b, "n": len(g), **m})
    return pd.DataFrame(rows)


def validate_confidence_flag(y_true_currency, y_pred_currency, flagged: np.ndarray) -> dict:
    flagged = np.asarray(flagged, dtype=bool)
    out = {}
    for label, mask in (("flagged", flagged), ("not_flagged", ~flagged)):
        if mask.sum() == 0:
            out[label] = {"n": 0}
            continue
        m = no_range_point_metrics(y_true_currency[mask], y_pred_currency[mask])
        out[label] = {"n": int(mask.sum()), **m}
    return out


def write_report(market: MarketConfig, sections: dict, path) -> None:
    lines = [f"# Evaluation Report — {market.name} ({market.currency_code})\n"]
    for title, content in sections.items():
        lines.append(f"## {title}\n")
        if isinstance(content, pd.DataFrame):
            lines.append(content.to_markdown(index=False))
        else:
            lines.append(str(content))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")
