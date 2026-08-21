"""Per-house SHAP explanations, uniform across model types.

TreeExplainer is used for XGBoost/RandomForest (exact, fast); LinearExplainer
for the Ridge baseline (exact, fast) -- KernelExplainer is deliberately
avoided since it's slow/approximate and unnecessary given the other two
cover every model this project trains.

SHAP runs on the *post-preprocessing* feature matrix (one-hot / ordinal /
target-encoded numbers), then contributions from encoded columns are
aggregated back to their original field name (e.g. all `ms_zoning_*`
one-hot columns collapse back into one "ms_zoning" contribution) so the
app shows "Zoning" rather than "nom__ms_zoning_RL".

Because the target is log-transformed, a SHAP value phi (in log-price
units) is converted to a human-readable multiplicative percentage impact:
(exp(phi) - 1) * 100 -- the technically correct reading of a log-target
SHAP contribution, not a naive linear currency amount.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.explain.feature_names import feature_name_map as _feature_name_map
from src.features.common import FeatureSpec

BACKGROUND_SAMPLE_SIZE = 100


@dataclass
class ExplanationResult:
    contributions_pct: dict[str, float]   # original_col -> multiplicative % impact
    base_value_log: float
    top_positive: list[tuple[str, float]]
    top_negative: list[tuple[str, float]]


class MarketExplainer:
    def __init__(self, fitted_pipeline: Pipeline, spec: FeatureSpec, X_background_raw: pd.DataFrame):
        self.pipeline = fitted_pipeline
        self.preprocessor = fitted_pipeline.named_steps["prep"]
        self.regressor = fitted_pipeline.named_steps["reg"]
        self.spec = spec
        self.feature_map = _feature_name_map(self.preprocessor, spec)

        bg = X_background_raw.sample(
            n=min(BACKGROUND_SAMPLE_SIZE, len(X_background_raw)), random_state=0
        )
        bg_transformed = self.preprocessor.transform(bg)
        self.background_transformed = bg_transformed

        if isinstance(self.regressor, (XGBRegressor, RandomForestRegressor)):
            self.explainer = shap.TreeExplainer(self.regressor)
            self._kind = "tree"
        elif isinstance(self.regressor, Ridge):
            self.explainer = shap.LinearExplainer(self.regressor, bg_transformed)
            self._kind = "linear"
        else:
            # Fallback: still works for the HistGBR quantile models if ever
            # explained directly, via the generic (slower) explainer.
            self.explainer = shap.Explainer(self.regressor, bg_transformed)
            self._kind = "generic"

    def explain(self, x_raw: pd.DataFrame) -> ExplanationResult:
        """x_raw: single-row DataFrame with the raw (pre-preprocessing) feature columns."""
        x_transformed = self.preprocessor.transform(x_raw)
        if self._kind == "tree":
            sv = self.explainer.shap_values(x_transformed)
            base_value = float(np.ravel(self.explainer.expected_value)[0])
        else:
            sv = self.explainer.shap_values(x_transformed)
            base_value = float(np.ravel(self.explainer.expected_value)[0])

        sv = np.asarray(sv).reshape(-1)
        transformed_names = self.preprocessor.get_feature_names_out()

        agg: dict[str, float] = {}
        for name, val in zip(transformed_names, sv):
            orig = self.feature_map.get(name, name)
            agg[orig] = agg.get(orig, 0.0) + float(val)

        pct = {k: (np.expm1(v) * 100) for k, v in agg.items()}
        ranked = sorted(pct.items(), key=lambda kv: kv[1], reverse=True)
        top_pos = [kv for kv in ranked if kv[1] > 0][:5]
        top_neg = sorted([kv for kv in ranked if kv[1] < 0], key=lambda kv: kv[1])[:5]

        return ExplanationResult(
            contributions_pct=pct,
            base_value_log=base_value,
            top_positive=top_pos,
            top_negative=top_neg,
        )
