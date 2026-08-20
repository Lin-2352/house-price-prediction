"""Clean the Ames Housing dataset (De Cock, 2011).

Source file uses De Cock's original spaced column names (e.g. "Gr Liv
Area"), not the Kaggle-competition-style names most tutorials assume.
Step 1 here is normalizing to snake_case so every downstream module has one
stable naming convention.

Key domain facts encoded here (see De Cock 2011 and the accompanying data
documentation):
  * A blank in most quality/type/finish columns means the structural
    feature doesn't exist (no basement / no garage / no fireplace / no
    pool / no fence / no alley access), not that the value is unknown.
    These become an explicit "None" category, not NaN, so they aren't
    silently dropped or mis-imputed later.
  * Numeric structural companions of those same features (e.g. Garage
    Cars/Area when there is no garage) are filled with 0.
  * De Cock's documented outlier rule: houses with Gr Liv Area > 4000 are
    known data-entry/partial-sale oddities and should be removed before
    modeling. This MUST happen before the train/cal/test split so the
    three splits remain drawn from the same population (conformal
    prediction's validity guarantee assumes exchangeability).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import DATA_RAW

RAW_FILE = DATA_RAW / "ames" / "AmesHousing.xls"

# Columns where NaN structurally means "this feature doesn't exist" -> "None"
NONE_MEANS_ABSENT = [
    "Alley", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Fireplace Qu", "Garage Type", "Garage Finish",
    "Garage Qual", "Garage Cond", "Pool QC", "Fence", "Misc Feature",
    "Mas Vnr Type",
]

# Numeric companions of absent structural features -> 0
ZERO_MEANS_ABSENT = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath",
    "Garage Cars", "Garage Area",
]

# Identifier / non-predictive columns to drop
DROP_COLS = ["Order", "PID"]

OUTLIER_GR_LIV_AREA = 4000


def _to_snake(col: str) -> str:
    col = col.strip()
    col = re.sub(r"[^0-9a-zA-Z]+", " ", col)  # split on non-alnum
    col = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", col)  # camelCase -> words
    return "_".join(col.lower().split())


def load_raw() -> pd.DataFrame:
    return pd.read_excel(RAW_FILE)


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_raw()
    df = df.copy()
    df.columns = [_to_snake(c) for c in df.columns]

    drop = [_to_snake(c) for c in DROP_COLS]
    df = df.drop(columns=[c for c in drop if c in df.columns])

    for c in NONE_MEANS_ABSENT:
        c = _to_snake(c)
        if c in df.columns:
            df[c] = df[c].fillna("None")

    for c in ZERO_MEANS_ABSENT:
        c = _to_snake(c)
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Garage Yr Blt: no garage -> no meaningful year; use 0 as an explicit
    # sentinel (kept out of any "age" arithmetic when garage_type == "None").
    gyb = _to_snake("Garage Yr Blt")
    if gyb in df.columns:
        df[gyb] = df[gyb].fillna(0)

    # Lot Frontage: genuinely missing (no structural meaning) -> leave NaN,
    # imputed later inside the modeling pipeline (median imputer), not here.

    # Drop duplicate rows if any (defensive; Ames is not known to have dupes)
    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        print(f"Dropped {before - len(df)} exact duplicate rows")

    # De Cock's documented outlier rule
    gla = _to_snake("Gr Liv Area")
    before = len(df)
    df = df[df[gla] <= OUTLIER_GR_LIV_AREA].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with {gla} > {OUTLIER_GR_LIV_AREA} (De Cock outlier rule)")

    sale_price = _to_snake("SalePrice")
    df = df.rename(columns={sale_price: "sale_price"})

    return df


if __name__ == "__main__":
    out = clean()
    print(out.shape)
    print(out.isna().sum().sort_values(ascending=False).head(10))
    from src.config import DATA_INTERIM
    out.to_parquet(DATA_INTERIM / "ames_clean.parquet", index=False)
    print(f"Saved to {DATA_INTERIM / 'ames_clean.parquet'}")
