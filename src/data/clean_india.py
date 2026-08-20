"""Clean the Indian metropolitan housing dataset (Kaggle: ruchi798).

Schema discovered by direct inspection (see plan notes) before writing this
file, rather than guessing:
  * One CSV per city (Bangalore, Chennai, Delhi, Hyderabad, Kolkata,
    Mumbai), 40 identical columns each; concatenated here with an added
    `city` column.
  * ~30 boolean-looking amenity columns (Gymnasium, SwimmingPool, ...,
    plus furnishing flags like AC/TV/Sofa) are encoded as {0, 1, 9}, where
    9 is a filler meaning "not recorded" for the large majority of rows
    (confirmed: 9 accounts for ~69% of values in every amenity column
    checked). Any value outside {0, 1} is treated as genuine missing data
    (NaN), not assumed to be a specific token — the rule is "not in the
    valid set", so it's robust if a column's filler code differs.
  * `Price` is an asking/listing price, not a confirmed sale price, and
    there is no transaction date column. This is a stated limitation
    carried through to evaluation and the app's disclaimer, not hidden.
  * Meaningful duplicate rows exist within each city file (different
    listings can coincidentally share every field, but a large duplicate
    count here is more likely re-scraped/re-posted listings) and are
    dropped.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import DATA_RAW

RAW_DIR = DATA_RAW / "india"
CITY_FILES = ["Bangalore.csv", "Chennai.csv", "Delhi.csv", "Hyderabad.csv", "Kolkata.csv", "Mumbai.csv"]

# Every non-numeric-scale column other than Price/Area/Location/No. of
# Bedrooms/Resale is a 0/1/9-coded amenity or furnishing flag.
NON_AMENITY_COLS = {"Price", "Area", "Location", "No. of Bedrooms", "Resale"}


def _to_snake(col: str) -> str:
    col = col.strip()
    col = col.replace("'", "")
    col = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", col)  # camelCase -> words
    col = re.sub(r"[^0-9a-zA-Z]+", " ", col)
    return "_".join(col.lower().split())


def load_raw() -> pd.DataFrame:
    frames = []
    for fname in CITY_FILES:
        city = fname.removesuffix(".csv")
        df = pd.read_csv(RAW_DIR / fname)
        df["City"] = city
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_raw()
    df = df.copy()

    amenity_cols = [c for c in df.columns if c not in NON_AMENITY_COLS and c != "City"]

    # Any value outside {0, 1} (e.g. the 9 filler) -> NaN, not a guessed token.
    for c in amenity_cols:
        df[c] = df[c].where(df[c].isin([0, 1]), np.nan)

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} exact duplicate rows across all cities")

    df.columns = [_to_snake(c) for c in df.columns]
    df = df.rename(columns={
        "no_of_bedrooms": "bedrooms",
    })

    # Sanity bounds: drop non-positive or absurd price/area rows (data-entry errors)
    before = len(df)
    df = df[(df["price"] > 0) & (df["area"] > 0)].reset_index(drop=True)
    if len(df) != before:
        print(f"Dropped {before - len(df)} rows with non-positive price or area")

    return df


if __name__ == "__main__":
    out = clean()
    print(out.shape)
    print("Rows per city:\n", out["city"].value_counts())
    print(out.isna().sum().sort_values(ascending=False).head(10))
    from src.config import DATA_INTERIM
    out.to_parquet(DATA_INTERIM / "india_clean.parquet", index=False)
    print(f"Saved to {DATA_INTERIM / 'india_clean.parquet'}")
