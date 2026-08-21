"""Inference logic for the FastAPI service -- a thin wrapper around the
same `src/` code the pipeline and Streamlit app already use. No new ML
logic lives here; this only adapts it to a JSON request/response shape.

Feature contributions are computed via XGBoost's native
`Booster.predict(..., pred_contribs=True)` rather than `shap.TreeExplainer`
-- mathematically identical output for tree ensembles (both compute exact
SHAP values via the same TreeSHAP algorithm), but avoids `shap` as a
runtime dependency for this service (see src/models/persist.py's
`save_api_bundle` docstring for why that matters on a memory-constrained
deployment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import get_market
from src.explain.feature_names import feature_name_map
from src.features.common import log1p_columns
from src.models import persist
from src.uncertainty.conformal import predict_interval_currency

MARKET_NAME = "india"


class Model:
    """Loaded once at process startup (see api/main.py's lifespan) and
    reused across requests -- loading the joblib bundle per-request would
    be needlessly slow and would repeatedly pay deserialization cost."""

    def __init__(self):
        market = get_market(MARKET_NAME)
        self.market = market
        self.bundle = persist.load_api_bundle(market.artifact_dir)
        self.metrics = _load_metrics(market)
        self.feature_map = feature_name_map(
            self.bundle["point_model"].named_steps["prep"], self.bundle["spec"]
        )

    def schema(self) -> dict:
        return {
            "form_schema": self.bundle["form_schema"],
            "important_fields": self.bundle["important_fields"],
            "currency_symbol": self.bundle["currency_symbol"],
            "currency_code": self.bundle["currency_code"],
            "point_model_name": self.bundle["point_model_name"],
            "test_mape": self.bundle["test_mape"],
            "test_rmse_log": self.bundle["test_rmse_log"],
        }

    def metrics_payload(self) -> dict:
        return self.metrics

    def predict(self, values: dict, range_method: str) -> dict:
        bundle = self.bundle
        schema = bundle["form_schema"]

        # Build a single-row DataFrame in the exact column order/types the
        # trained pipeline expects; missing/None -> NaN (same as the
        # Streamlit app's "skip" checkbox path -- SimpleImputer inside the
        # pipeline handles it identically).
        row = {}
        skipped_important = []
        important_set = set(bundle["important_fields"])
        for col, meta in schema.items():
            val = values.get(col)
            if val is None or val == "":
                row[col] = np.nan
                if col in important_set:
                    skipped_important.append(col)
                continue
            row[col] = float(val) if meta["type"] == "numeric" else str(val)
        X_row = pd.DataFrame([row])
        X_row_log = log1p_columns(X_row, bundle["log_feature_cols"])

        model = bundle["point_model"]
        point_log = model.predict(X_row_log)
        point_currency = float(np.expm1(point_log[0]))

        conformal_obj = bundle["cqr"] if range_method == "cqr" else bundle["split_conformal"]
        _, lo, hi = predict_interval_currency(conformal_obj, X_row_log)
        lower, upper = float(lo[0]), float(hi[0])

        flagger = bundle["confidence_flagger"]
        prep = model.named_steps["prep"]
        X_transformed = prep.transform(X_row_log)
        novelty = flagger.novelty_score(X_transformed)[0]
        width_ratio = (upper - lower) / max(point_currency, 1)
        missing_count = int(flagger.missing_field_count(X_row_log)[0])
        is_flagged = bool(
            flagger.flag(np.array([width_ratio]), np.array([novelty]), np.array([missing_count]))[0]
        )
        reasons = flagger.explain_flag(width_ratio, novelty, missing_count)

        contributions = self._contributions(model, X_transformed)

        return {
            "predicted_price": point_currency,
            "range_low": lower,
            "range_high": upper,
            "range_method": range_method,
            "flagged": is_flagged,
            "flag_reasons": reasons,
            "skipped_important": skipped_important,
            "contributions": contributions,
        }

    def _contributions(self, model, X_transformed: np.ndarray) -> list[dict]:
        booster = model.named_steps["reg"].get_booster()
        dmatrix = xgb.DMatrix(X_transformed)
        contribs = booster.predict(dmatrix, pred_contribs=True)[0]
        values = contribs[:-1]  # last column is the bias/base-value term

        transformed_names = model.named_steps["prep"].get_feature_names_out()
        agg: dict[str, float] = {}
        for name, val in zip(transformed_names, values):
            orig = self.feature_map.get(name, name)
            agg[orig] = agg.get(orig, 0.0) + float(val)

        pct = {k: float(np.expm1(v) * 100) for k, v in agg.items()}
        ranked = sorted(pct.items(), key=lambda kv: kv[1], reverse=True)
        top_pos = [{"feature": k, "pct_impact": v} for k, v in ranked if v > 0][:5]
        top_neg = sorted(
            [{"feature": k, "pct_impact": v} for k, v in ranked if v < 0],
            key=lambda d: d["pct_impact"],
        )[:5]
        return top_pos + top_neg


def _load_metrics(market) -> dict:
    import json
    path = market.artifact_dir / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))
