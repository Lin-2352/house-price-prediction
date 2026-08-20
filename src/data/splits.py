"""Three-way train / calibration / test split, per market.

The calibration split is touched only when fitting conformal predictors
(never used to fit or tune the point-prediction models) and the test split
is touched exactly once, at final evaluation. Splits are stratified by a
coarse price-quartile bucket so each split has a comparable price mix.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CAL_FRAC, RANDOM_SEED, TEST_FRAC, TRAIN_FRAC, MarketConfig


def make_splits(df: pd.DataFrame, market: MarketConfig) -> dict[str, pd.DataFrame]:
    assert abs(TRAIN_FRAC + CAL_FRAC + TEST_FRAC - 1.0) < 1e-9

    strata = pd.qcut(df[market.price_col], q=5, duplicates="drop")

    train_df, rest_df, strata_train, strata_rest = train_test_split(
        df, strata, test_size=(CAL_FRAC + TEST_FRAC),
        stratify=strata, random_state=RANDOM_SEED,
    )
    rel_test_frac = TEST_FRAC / (CAL_FRAC + TEST_FRAC)
    cal_df, test_df = train_test_split(
        rest_df, test_size=rel_test_frac,
        stratify=strata_rest, random_state=RANDOM_SEED,
    )

    for name, part in (("train", train_df), ("calibration", cal_df), ("test", test_df)):
        print(f"{market.name}/{name}: {len(part)} rows ({len(part) / len(df):.1%})")

    return {
        "train": train_df.reset_index(drop=True),
        "calibration": cal_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
