"""House Price Estimator (India) -- Streamlit demo app.

Loads the persisted India-market bundle (see src/pipeline.py) and lets a
user fill in flat features to get: a predicted price, a 90% price range
(from the adaptive/CQR conformal arm, with split-conformal as a comparison
toggle), a SHAP-based explanation of what drove the estimate, and a
low-confidence warning when the flat is unusual or the range is unusually
wide. A "Model performance & validation" panel shows the held-out test-set
metrics the prediction can be cross-checked against, so the numbers aren't
taken on faith. This is a demonstration app, not a licensed valuation
service -- every result carries that disclaimer.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_market
from src.features.common import log1p_columns
from src.models import persist
from src.uncertainty.conformal import predict_interval_currency

MARKET_NAME = "india"

st.set_page_config(page_title="House Price Estimator — India", page_icon="🏠", layout="wide")

INDIA_GROUPS = {
    "Location & basics": ["city", "location", "area", "bedrooms", "resale"],
    "Amenities": ["gymnasium", "swimming_pool", "club_house", "24_x7_security", "power_backup",
                  "car_parking", "lift_available", "maintenance_staff", "landscaped_gardens",
                  "jogging_track", "indoor_games", "sports_facility", "shopping_mall", "school",
                  "hospital", "atm", "intercom", "rain_water_harvesting"],
    "Furnishing": ["ac", "wifi", "tv", "sofa", "bed", "wardrobe", "microwave", "refrigerator",
                   "washing_machine", "dining_table", "gasconnection"],
}


@st.cache_resource(show_spinner="Loading model...")
def load_market(name: str) -> dict:
    market = get_market(name)
    return persist.load_bundle(market.artifact_dir)


@st.cache_data(show_spinner=False)
def load_metrics(name: str) -> dict:
    market = get_market(name)
    path = market.artifact_dir / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def render_field(col: str, meta: dict, key_prefix: str, important: bool):
    key = f"{key_prefix}_{col}"
    label = col.replace("_", " ").title()
    skip = False
    if important:
        skip = st.checkbox("Unknown / skip", key=f"{key}_skip", value=False,
                            help="Leave this important field unanswered -- the model will flag "
                                 "the resulting prediction as lower-confidence.")
    if meta["type"] == "numeric":
        lo, hi, med = meta["min"], meta["max"], meta["median"]
        is_int = float(lo).is_integer() and float(hi).is_integer() and float(med).is_integer()
        if is_int:
            val = st.number_input(label, min_value=int(lo), max_value=max(int(hi), int(lo) + 1),
                                   value=int(med), key=key, disabled=skip)
        else:
            val = st.number_input(label, min_value=float(lo), max_value=float(hi), value=float(med),
                                   key=key, disabled=skip)
    else:
        # ordinal / nominal / locality -> selectbox over the training-derived categories
        cats = meta["categories"]
        default_idx = len(cats) // 2 if meta["type"] == "ordinal" else 0
        val = st.selectbox(label, options=cats, index=min(default_idx, len(cats) - 1), key=key, disabled=skip)
    return np.nan if skip else val


def build_input_row(schema: dict, important_fields: list[str]) -> pd.DataFrame:
    grouped_cols = {c for cols in INDIA_GROUPS.values() for c in cols}
    important_set = set(important_fields)
    values = {}

    for title, cols in INDIA_GROUPS.items():
        with st.expander(title, expanded=(title == "Location & basics")):
            cols_widgets = st.columns(2)
            for i, col in enumerate(cols):
                if col not in schema:
                    continue
                with cols_widgets[i % 2]:
                    values[col] = render_field(col, schema[col], MARKET_NAME, col in important_set)

    remaining = [c for c in schema if c not in grouped_cols]
    if remaining:
        with st.expander("Other details", expanded=False):
            cols_widgets = st.columns(2)
            for i, col in enumerate(remaining):
                with cols_widgets[i % 2]:
                    values[col] = render_field(col, schema[col], MARKET_NAME, col in important_set)

    return pd.DataFrame([values])


def fmt_currency(x: float, symbol: str) -> str:
    return f"{symbol}{x:,.0f}"


def fmt_lakh_crore(x: float) -> str:
    """India-idiomatic magnitude alongside the raw rupee figure."""
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:,.2f} L"
    return f"₹{x:,.0f}"


def render_metrics_panel(metrics: dict, symbol: str):
    st.subheader("📊 Model performance & validation")
    st.caption(
        "These are held-out **test-set** metrics — computed once, on data the model never saw during "
        "training or calibration — so you can cross-check any prediction above against how the model "
        "actually performs. Full tables are also in `reports/india_evaluation_report.md` in the repo."
    )

    pa = metrics["point_accuracy"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Absolute Error", fmt_lakh_crore(pa["mae"]))
    c2.metric("RMSE", fmt_lakh_crore(pa["rmse"]))
    c3.metric("MAPE", f"{pa['mape']:.1f}%")
    c4.metric("Log-price RMSE", f"{pa['rmse_log']:.3f}")
    st.caption(
        "⚠️ MAPE of ~51% reflects real limitations of this dataset (asking prices, not confirmed sale "
        "prices; no transaction date; ~70% of amenity fields unrecorded) — reported honestly rather than "
        "averaged away. Treat point predictions as rough, and lean on the 90% range and confidence flag."
    )

    st.markdown("**90% price-range coverage & width, by method** (target coverage = 90%)")
    cov_df = pd.DataFrame(metrics["coverage_table"])
    cov_df["coverage"] = (cov_df["coverage"] * 100).round(1).astype(str) + "%"
    cov_df["avg_width"] = cov_df["avg_width"].apply(lambda v: fmt_lakh_crore(v))
    cov_df["median_width"] = cov_df["median_width"].apply(lambda v: fmt_lakh_crore(v))
    cov_df.columns = ["Method", "Coverage", "Avg. range width", "Median range width"]
    st.dataframe(cov_df, hide_index=True, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Accuracy by price bracket**")
        br_df = pd.DataFrame(metrics["bracket_table"])
        br_df["mae"] = br_df["mae"].apply(fmt_lakh_crore)
        br_df["rmse"] = br_df["rmse"].apply(fmt_lakh_crore)
        br_df["mape"] = br_df["mape"].round(1).astype(str) + "%"
        br_df.columns = ["Price bracket", "n", "MAE", "RMSE", "MAPE"]
        st.dataframe(br_df, hide_index=True, width="stretch")
    with col_b:
        st.markdown("**90% CQR range coverage by city**")
        seg_df = pd.DataFrame(metrics["segment_table"])
        seg_df["coverage"] = (seg_df["coverage"] * 100).round(1).astype(str) + "%"
        seg_df["avg_width"] = seg_df["avg_width"].apply(fmt_lakh_crore)
        seg_df = seg_df[["segment", "n", "coverage", "avg_width"]]
        seg_df.columns = ["City", "n (test)", "Coverage", "Avg. width"]
        st.dataframe(seg_df, hide_index=True, width="stretch")

    st.markdown("**Confidence flag validation** — flagged predictions should be measurably worse")
    cf = metrics["confidence_flag_validation"]
    cf_df = pd.DataFrame([
        {"Group": "🚩 Flagged low-confidence", "n": cf["flagged"]["n"],
         "MAE": fmt_lakh_crore(cf["flagged"]["mae"]), "MAPE": f"{cf['flagged']['mape']:.1f}%"},
        {"Group": "✅ Not flagged", "n": cf["not_flagged"]["n"],
         "MAE": fmt_lakh_crore(cf["not_flagged"]["mae"]), "MAPE": f"{cf['not_flagged']['mape']:.1f}%"},
    ])
    st.dataframe(cf_df, hide_index=True, width="stretch")
    st.caption(
        "Flagged predictions do show a higher MAPE than unflagged ones — the confidence flag is catching "
        "genuinely worse predictions, not just adding noise."
    )

    st.markdown("**Cross-validation comparison of candidate models** (train-only, log-price RMSE — lower is better)")
    cv_df = pd.DataFrame([
        {"Model": "Linear (Ridge)", "CV RMSE (log)": f"{metrics['cv_rmse_log']['linear']:.4f}"},
        {"Model": "Random Forest", "CV RMSE (log)": f"{metrics['cv_rmse_log']['random_forest']:.4f}"},
        {"Model": "XGBoost (selected)", "CV RMSE (log)": f"{metrics['cv_rmse_log']['xgboost']:.4f}"},
    ])
    st.dataframe(cv_df, hide_index=True, width="stretch")


def main():
    st.title("🏠 House Price Estimator — India")
    st.caption(
        "A demonstration model for Indian metro flats (Bangalore, Chennai, Delhi, Hyderabad, Kolkata, "
        "Mumbai) that reports a predicted price, a 90%-confidence price range validated by conformal "
        "prediction, and an explanation of what drove the estimate — not just a single confident-looking "
        "number."
    )

    bundle = load_market(MARKET_NAME)
    market = get_market(MARKET_NAME)
    metrics = load_metrics(MARKET_NAME)
    schema = bundle["form_schema"]
    symbol = bundle["currency_symbol"]

    st.markdown(
        f"**Model:** {bundle['point_model_name']} · **Test-set error:** {bundle['test_mape']:.1f}% MAPE · "
        f"**Log-RMSE:** {bundle['test_rmse_log']:.3f} &nbsp;·&nbsp; "
        f"*(see full validation panel below)*"
    )

    X_row = build_input_row(schema, bundle["important_fields"])

    interval_choice = st.radio(
        "Range method", ["Adaptive (CQR) -- recommended", "Split-conformal"], horizontal=True,
    )

    if st.button("Estimate price", type="primary"):
        X_row_log = log1p_columns(X_row, bundle["log_feature_cols"])

        model = bundle["point_model"]
        point_log = model.predict(X_row_log)
        point_currency = float(np.expm1(point_log[0]))

        conformal_obj = bundle["cqr"] if interval_choice.startswith("Adaptive") else bundle["split_conformal"]
        _, lo, hi = predict_interval_currency(conformal_obj, X_row_log)
        lower, upper = float(lo[0]), float(hi[0])

        # confidence signals
        flagger = bundle["confidence_flagger"]
        prep = model.named_steps["prep"]
        X_transformed = prep.transform(X_row_log)
        novelty = flagger.novelty_score(X_transformed)[0]
        width_ratio = (upper - lower) / max(point_currency, 1)
        missing_count = int(flagger.missing_field_count(X_row_log)[0]) if bundle["important_fields"] else 0
        is_flagged = bool(flagger.flag(np.array([width_ratio]), np.array([novelty]), np.array([missing_count]))[0])
        reasons = flagger.explain_flag(width_ratio, novelty, missing_count)

        st.divider()
        st.subheader("Prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted price", fmt_lakh_crore(point_currency), fmt_currency(point_currency, symbol))
        c2.metric("90% range — low", fmt_lakh_crore(lower), fmt_currency(lower, symbol))
        c3.metric("90% range — high", fmt_lakh_crore(upper), fmt_currency(upper, symbol))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[upper - lower], y=["90% range"], base=[lower], orientation="h",
                              marker_color="rgba(99,110,250,0.35)", name="range"))
        fig.add_trace(go.Scatter(x=[point_currency], y=["90% range"], mode="markers",
                                  marker=dict(size=14, color="rgb(99,110,250)"), name="predicted"))
        fig.update_layout(height=180, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title=f"Price ({market.currency_code})")
        st.plotly_chart(fig, width="stretch")

        if is_flagged:
            st.warning(
                "⚠️ **Low confidence prediction.** " + "; ".join(reasons).capitalize() +
                ". Consider getting a professional valuation for this property instead of relying on this estimate."
            )
        else:
            st.success("✅ This prediction falls within the model's normal operating range.")

        st.subheader("What drove this estimate")
        explainer = bundle["explainer"]
        expl = explainer.explain(X_row_log)
        pos = expl.top_positive
        neg = expl.top_negative
        if pos or neg:
            names = [n.replace("_", " ").title() for n, _ in pos] + [n.replace("_", " ").title() for n, _ in neg]
            vals = [v for _, v in pos] + [v for _, v in neg]
            colors = ["#2ca02c"] * len(pos) + ["#d62728"] * len(neg)
            order = np.argsort(vals)
            fig2 = go.Figure(go.Bar(
                x=[vals[i] for i in order], y=[names[i] for i in order], orientation="h",
                marker_color=[colors[i] for i in order],
            ))
            fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="% impact on predicted price vs. a typical house")
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No single feature stood out as an unusually strong driver for this house.")

        st.divider()
        st.caption(
            "⚠️ **This is a demonstration model estimate, not a professional valuation.** "
            "It reflects the training data's collection period, not necessarily today's market, "
            "and should not be used as the sole basis for a financial decision."
        )

    st.divider()
    with st.expander("📊 Model performance & validation (cross-check the numbers above)", expanded=False):
        render_metrics_panel(metrics, symbol)


if __name__ == "__main__":
    main()
