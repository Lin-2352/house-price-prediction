"""Shared preprocessing-pipeline builder used by both markets.

One ColumnTransformer handles four kinds of columns uniformly:
  * numeric              -> median impute + standard-scale
  * ordinal (domain map)  -> OrdinalEncoder with an explicit, hand-specified
                             category order (never alphabetical auto-order)
  * nominal (low card.)   -> one-hot encode
  * locality (high card.) -> sklearn TargetEncoder (internally cross-fitted,
                             so fitting it on the training split alone does
                             not leak that split's own target into itself)

Using one shared pipeline builder means the same "aggregate encoded columns
back to their original field name" logic in explain/shap_explainer.py works
identically for both markets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from src.config import RANDOM_SEED


class RareCategoryCollapser(BaseEstimator, TransformerMixin):
    """Map categories seen fewer than `min_count` times in the fit data (or
    never seen at all) to `other_label`. Fit strictly on the training
    fold, so calibration/test rows never influence which categories count
    as "rare" -- avoiding leakage into the locality encoding that follows.
    """

    def __init__(self, min_count: int = 0, other_label: str = "other"):
        self.min_count = min_count
        self.other_label = other_label

    def fit(self, X, y=None):
        s = pd.DataFrame(X).iloc[:, 0]
        if self.min_count > 0:
            counts = s.value_counts()
            self.keep_ = set(counts[counts >= self.min_count].index)
        else:
            self.keep_ = set(s.unique())
        return self

    def transform(self, X):
        s = pd.DataFrame(X).iloc[:, 0]
        out = s.where(s.isin(self.keep_), self.other_label)
        return out.to_frame()

    def get_feature_names_out(self, input_features=None):
        # Passes the single column's name through unchanged -- this step
        # only recodes rare values, it doesn't add/remove/rename columns.
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(["x0"], dtype=object)


@dataclass
class FeatureSpec:
    numeric_cols: list[str]
    ordinal_specs: dict[str, list[str]]   # column -> ordered categories, low to high
    nominal_cols: list[str]
    locality_col: str
    rare_locality_threshold: int = 0      # localities with fewer train rows than this -> "other"


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    ordinal_cols = list(spec.ordinal_specs.keys())
    ordinal_categories = [spec.ordinal_specs[c] for c in ordinal_cols]
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    locality_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="other")),
        ("collapse_rare", RareCategoryCollapser(min_count=spec.rare_locality_threshold)),
        ("encode", TargetEncoder(random_state=RANDOM_SEED)),
    ])

    transformers = [
        ("num", numeric_pipe, spec.numeric_cols),
        ("ord", ordinal_pipe, ordinal_cols),
        ("nom", nominal_pipe, spec.nominal_cols),
        ("loc", locality_pipe, [spec.locality_col]),
    ]
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def log1p_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of df with the given columns log1p-transformed.

    Applied to price and floor-area, both heavily right-skewed in both
    markets. Safe to call on train/cal/test independently since log1p is a
    fixed, data-independent transform (no fitting involved, no leakage).
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = np.log1p(df[c].astype(float))
    return df


def inverse_log1p(x):
    return np.expm1(x)


def build_form_schema(X_train: pd.DataFrame, spec: FeatureSpec) -> dict:
    """Small, JSON/joblib-friendly description of each input field's valid
    range/categories, built from the training split only. Lets the
    Streamlit app render a form and validate input without shipping the
    full training set into the deployed bundle.
    """
    schema: dict[str, dict] = {}

    for col in spec.numeric_cols:
        s = X_train[col].dropna()
        schema[col] = {
            "type": "numeric",
            "min": float(s.min()) if len(s) else 0.0,
            "max": float(s.max()) if len(s) else 1.0,
            "median": float(s.median()) if len(s) else 0.0,
        }

    for col, categories in spec.ordinal_specs.items():
        schema[col] = {"type": "ordinal", "categories": list(categories)}

    for col in spec.nominal_cols:
        cats = sorted(X_train[col].dropna().unique().tolist())
        schema[col] = {"type": "nominal", "categories": cats}

    loc_col = spec.locality_col
    counts = X_train[loc_col].value_counts()
    if spec.rare_locality_threshold > 0:
        keep = sorted(counts[counts >= spec.rare_locality_threshold].index.tolist())
    else:
        keep = sorted(counts.index.tolist())
    schema[loc_col] = {"type": "locality", "categories": keep + ["other"]}

    return schema
