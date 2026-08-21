"""Maps post-preprocessing transformed column names back to their original
input column names (e.g. all `nom__ms_zoning_RL`-style one-hot columns
collapse back to "ms_zoning"). Used to aggregate per-column contributions
-- whether from SHAP (src/explain/shap_explainer.py, used offline by the
pipeline/Streamlit app) or from XGBoost's native `pred_contribs` (used by
the deployed API, see api/inference.py) -- back to human-readable feature
names.

Deliberately has NO dependency on `shap` (numpy/pandas only) so the
deployed API can import it without pulling in shap's numba/llvmlite import
weight just to unpickle a bundle that doesn't even carry a SHAP object.
"""
from __future__ import annotations

from src.features.common import FeatureSpec


def feature_name_map(preprocessor, spec: FeatureSpec) -> dict[str, str]:
    names = preprocessor.get_feature_names_out()
    nominal_sorted = sorted(spec.nominal_cols, key=len, reverse=True)
    mapping = {}
    for full_name in names:
        group, _, remainder = full_name.partition("__")
        if group in ("num", "ord", "loc"):
            mapping[full_name] = remainder
        elif group == "nom":
            match = None
            for col in nominal_sorted:
                if remainder == col or remainder.startswith(col + "_"):
                    match = col
                    break
            mapping[full_name] = match or remainder
        else:
            mapping[full_name] = remainder
    return mapping
