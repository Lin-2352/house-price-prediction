"""Feature engineering for the Indian metropolitan housing market.

No transaction date exists in this data, so no age-style derived features
are possible (stated limitation, carried through to the report/app). The
~34 amenity/furnishing flags are 0/1 with a large share missing (the
original "9 = not recorded" filler, already converted to NaN in
clean_india.py); they're imputed with the column's training-fold median
(the shared numeric pipeline in features/common.py), never on the full
dataset, so calibration/test rows never influence the imputed value.
"""
from __future__ import annotations

import pandas as pd

from src.features.common import FeatureSpec

AMENITY_COLS = [
    "maintenance_staff", "gymnasium", "swimming_pool", "landscaped_gardens",
    "jogging_track", "rain_water_harvesting", "indoor_games", "shopping_mall",
    "intercom", "sports_facility", "atm", "club_house", "school",
    "24_x7_security", "power_backup", "car_parking", "staff_quarter",
    "cafeteria", "multipurpose_room", "hospital", "washing_machine",
    "gasconnection", "ac", "wifi", "childrensplayarea", "lift_available",
    "bed", "vaastu_compliant", "microwave", "golf_course", "tv",
    "dining_table", "sofa", "wardrobe", "refrigerator",
]

NUMERIC_COLS = ["area", "bedrooms", "resale"] + AMENITY_COLS

NOMINAL_COLS = ["city"]

LOCALITY_COL = "location"

# Localities (free-text, long tail) with fewer than this many training rows
# are collapsed to "other" before target encoding, so the encoding isn't
# driven by single-occurrence noise.
RARE_LOCALITY_MIN_COUNT = 15

# High-importance inputs for the confidence flag's "missing important
# fields" signal (see confidence/flag.py).
IMPORTANT_FIELDS = ["area", "bedrooms", "location", "city", "resale"]


def derive(df: pd.DataFrame) -> pd.DataFrame:
    # No date/age features possible for this dataset; kept as a no-op hook
    # so the pipeline shape mirrors ames_features.derive().
    return df.copy()


def feature_spec() -> FeatureSpec:
    return FeatureSpec(
        numeric_cols=NUMERIC_COLS,
        ordinal_specs={},
        nominal_cols=NOMINAL_COLS,
        locality_col=LOCALITY_COL,
        rare_locality_threshold=RARE_LOCALITY_MIN_COUNT,
    )
