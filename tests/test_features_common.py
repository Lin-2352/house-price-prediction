import numpy as np
import pandas as pd

from src.features.common import RareCategoryCollapser, inverse_log1p, log1p_columns


def test_log1p_roundtrip():
    df = pd.DataFrame({"price": [100.0, 200.0, 0.0]})
    logged = log1p_columns(df, ["price"])
    restored = inverse_log1p(logged["price"])
    assert np.allclose(restored, df["price"])


def test_log1p_does_not_mutate_input():
    df = pd.DataFrame({"price": [100.0]})
    log1p_columns(df, ["price"])
    assert df["price"].iloc[0] == 100.0


def test_rare_category_collapser_fits_on_given_data_only():
    train = pd.DataFrame({"loc": ["A"] * 10 + ["B"] * 2})
    collapser = RareCategoryCollapser(min_count=5)
    collapser.fit(train[["loc"]])

    test = pd.DataFrame({"loc": ["A", "B", "C"]})  # B rare, C unseen
    out = collapser.transform(test[["loc"]])
    assert out["loc"].tolist() == ["A", "other", "other"]
