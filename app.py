"""House Price Estimator -- Streamlit demo app.

Loads a persisted per-market bundle (see src/pipeline.py) and lets a user
fill in house features to get: a predicted price, a 90% price range (from
the adaptive/CQR conformal arm, with split-conformal as a comparison
toggle), a SHAP-based explanation of what drove the estimate, and a
low-confidence warning when the house is unusual or the range is unusually
wide. This is a demonstration app, not a licensed valuation service --
every result carries that disclaimer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_market
from src.features.common import log1p_columns
from src.models import persist
from src.uncertainty.conformal import predict_interval_currency

st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="wide")

MARKET_LABELS = {"ames": "🇺🇸 United States (Ames, Iowa)", "india": "🇮🇳 India (metro cities)"}

# Ames has 82 input fields -- grouped into collapsible sections so the form
# stays usable. Anything not listed falls into "Other details".
AMES_GROUPS = {
    "Location & basics": ["neighborhood", "ms_zoning", "lot_area", "lot_frontage", "bldg_type", "house_style"],
    "Size & layout": ["gr_liv_area", "total_area", "total_bsmt_sf", "1st_flr_sf", "2nd_flr_sf",
                       "bedroom_abv_gr", "full_bath", "half_bath", "kitchen_abv_gr", "tot_rms_abv_grd"],
    "Age & condition": ["house_age", "years_since_remodel", "overall_qual", "overall_cond",
                         "exter_qual", "exter_cond", "kitchen_qual", "heating_qc"],
    "Basement & garage": ["bsmt_qual", "bsmt_cond", "bsmt_exposure", "bsmt_fin_type_1", "bsmt_fin_sf_1",
                           "garage_type", "garage_finish", "garage_cars", "garage_area", "garage_age"],
    "Outdoor & extras": ["wood_deck_sf", "open_porch_sf", "pool_area", "fireplaces", "fireplace_qu",
                          "fence", "central_air", "paved_drive"],
}

INDIA_GROUPS = {
    "Location & basics": ["city", "location", "area", "bedrooms", "resale"],
    "Amenities": ["gymnasium", "swimming_pool", "club_house", "24_x7_security", "power_backup",
                  "car_parking", "lift_available", "maintenance_staff", "landscaped_gardens",
                  "jogging_track", "indoor_games", "sports_facility", "shopping_mall", "school",
                  "hospital", "atm", "intercom", "rain_water_harvesting"],
    "Furnishing": ["ac", "wifi", "tv", "sofa", "bed", "wardrobe", "microwave", "refrigerator",
                   "washing_machine", "dining_table", "gasconnection"],
}

GROUPS = {"ames": AMES_GROUPS, "india": INDIA_GROUPS}


@st.cache_resource(show_spinner="Loading model...")
def load_market(name: str) -> dict:
    market = get_market(name)
    return persist.load_bundle(market.artifact_dir)


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


def build_input_row(schema: dict, market_name: str, important_fields: list[str]) -> pd.DataFrame:
    groups = GROUPS[market_name]
    grouped_cols = {c for cols in groups.values() for c in cols}
    important_set = set(important_fields)
    values = {}

    for title, cols in groups.items():
        with st.expander(title, expanded=(title == "Location & basics")):
            cols_widgets = st.columns(2)
            for i, col in enumerate(cols):
                if col not in schema:
                    continue
                with cols_widgets[i % 2]:
                    values[col] = render_field(col, schema[col], market_name, col in important_set)

    remaining = [c for c in schema if c not in grouped_cols]
    if remaining:
        with st.expander("Other details", expanded=False):
            cols_widgets = st.columns(2)
            for i, col in enumerate(remaining):
                with cols_widgets[i % 2]:
                    values[col] = render_field(col, schema[col], market_name, col in important_set)

    return pd.DataFrame([values])


def fmt_currency(x: float, symbol: str) -> str:
    return f"{symbol}{x:,.0f}"


def main():
    st.title("🏠 House Price Estimator")
    st.caption(
        "A demonstration model that reports a predicted price, a 90%-confidence price range "
        "validated by conformal prediction, and an explanation of what drove the estimate -- "
        "not just a single confident-looking number."
    )

    market_name = st.radio("Market", options=list(MARKET_LABELS), format_func=lambda k: MARKET_LABELS[k], horizontal=True)
    bundle = load_market(market_name)
    market = get_market(market_name)
    schema = bundle["form_schema"]

    st.markdown(f"**Model:** {bundle['point_model_name']} · **Test-set error:** {bundle['test_mape']:.1f}% MAPE · "
                f"**Log-RMSE:** {bundle['test_rmse_log']:.3f}")

    X_row = build_input_row(schema, market_name, bundle["important_fields"])

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

        symbol = bundle["currency_symbol"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted price", fmt_currency(point_currency, symbol))
        c2.metric("90% range — low", fmt_currency(lower, symbol))
        c3.metric("90% range — high", fmt_currency(upper, symbol))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[upper - lower], y=["90% range"], base=[lower], orientation="h",
                              marker_color="rgba(99,110,250,0.35)", name="range"))
        fig.add_trace(go.Scatter(x=[point_currency], y=["90% range"], mode="markers",
                                  marker=dict(size=14, color="rgb(99,110,250)"), name="predicted"))
        fig.update_layout(height=180, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title=f"Price ({market.currency_code})")
        st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("No single feature stood out as an unusually strong driver for this house.")

        st.divider()
        st.caption(
            "⚠️ **This is a demonstration model estimate, not a professional valuation.** "
            f"It reflects the {market_name} training data's collection period, not necessarily today's market, "
            "and should not be used as the sole basis for a financial decision."
        )


if __name__ == "__main__":
    main()
