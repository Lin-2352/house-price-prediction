"""Feature engineering for the Ames Housing market.

`derive()` adds a small number of domain-derived features and explicitly
excludes any price-per-area style feature (that would be computed from the
target and would leak it). `FEATURE_SPEC` describes every input column's
role for src.features.common.build_preprocessor.
"""
from __future__ import annotations

import pandas as pd

from src.features.common import FeatureSpec

NA_LOW = "None"  # matches clean_ames.py's fill value for absent structural features

QUALITY_ORDER = [NA_LOW, "Po", "Fa", "TA", "Gd", "Ex"]

ORDINAL_SPECS: dict[str, list[str]] = {
    "exter_qual": QUALITY_ORDER,
    "exter_cond": QUALITY_ORDER,
    "bsmt_qual": QUALITY_ORDER,
    "bsmt_cond": QUALITY_ORDER,
    "heating_qc": QUALITY_ORDER,
    "kitchen_qual": QUALITY_ORDER,
    "fireplace_qu": QUALITY_ORDER,
    "garage_qual": QUALITY_ORDER,
    "garage_cond": QUALITY_ORDER,
    "pool_qc": QUALITY_ORDER,
    "bsmt_exposure": [NA_LOW, "No", "Mn", "Av", "Gd"],
    "bsmt_fin_type_1": [NA_LOW, "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "bsmt_fin_type_2": [NA_LOW, "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "garage_finish": [NA_LOW, "Unf", "RFn", "Fin"],
    "fence": [NA_LOW, "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "lot_shape": ["IR3", "IR2", "IR1", "Reg"],
    "land_slope": ["Sev", "Mod", "Gtl"],
    "paved_drive": ["N", "P", "Y"],
    "utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    "functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "central_air": ["N", "Y"],
}

NOMINAL_COLS = [
    "ms_zoning", "street", "alley", "land_contour", "lot_config",
    "condition_1", "condition_2", "bldg_type", "house_style", "roof_style",
    "roof_matl", "exterior_1st", "exterior_2nd", "mas_vnr_type",
    "foundation", "heating", "electrical", "garage_type", "misc_feature",
    "sale_type", "sale_condition",
]

NUMERIC_COLS = [
    "ms_sub_class", "lot_frontage", "lot_area", "overall_qual", "overall_cond",
    "mas_vnr_area", "bsmt_fin_sf_1", "bsmt_fin_sf_2", "bsmt_unf_sf",
    "total_bsmt_sf", "1st_flr_sf", "2nd_flr_sf", "low_qual_fin_sf",
    "gr_liv_area", "bsmt_full_bath", "bsmt_half_bath", "full_bath",
    "half_bath", "bedroom_abv_gr", "kitchen_abv_gr", "tot_rms_abv_grd",
    "fireplaces", "garage_cars", "garage_area", "wood_deck_sf",
    "open_porch_sf", "enclosed_porch", "3_ssn_porch", "screen_porch",
    "pool_area", "misc_val", "mo_sold", "yr_sold",
    # derived below
    "total_area", "house_age", "years_since_remodel", "garage_age",
]

LOCALITY_COL = "neighborhood"

# High-importance inputs for the confidence flag's "missing important
# fields" signal (see confidence/flag.py) -- matters most in the live app
# when a user leaves fields blank.
IMPORTANT_FIELDS = [
    "overall_qual", "gr_liv_area", "total_bsmt_sf", "garage_cars",
    "house_age", "neighborhood", "full_bath", "bedroom_abv_gr",
]


def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_area"] = df["gr_liv_area"] + df["total_bsmt_sf"]
    df["house_age"] = (df["yr_sold"] - df["year_built"]).clip(lower=0)
    df["years_since_remodel"] = (df["yr_sold"] - df["year_remod_add"]).clip(lower=0)
    # garage_yr_blt is 0 (sentinel) when there is no garage; age is only
    # meaningful when a garage exists, so we don't produce a bogus huge age.
    has_garage = df["garage_yr_blt"] > 0
    df["garage_age"] = 0.0
    df.loc[has_garage, "garage_age"] = (
        df.loc[has_garage, "yr_sold"] - df.loc[has_garage, "garage_yr_blt"]
    ).clip(lower=0)
    return df


def feature_spec() -> FeatureSpec:
    return FeatureSpec(
        numeric_cols=NUMERIC_COLS,
        ordinal_specs=ORDINAL_SPECS,
        nominal_cols=NOMINAL_COLS,
        locality_col=LOCALITY_COL,
    )
