import numpy as np

from src.confidence.flag import ConfidenceFlagger


def test_missing_field_count():
    import pandas as pd
    flagger = ConfidenceFlagger(important_fields=["a", "b", "c"])
    df = pd.DataFrame([{"a": 1, "b": np.nan, "c": 3}])
    counts = flagger.missing_field_count(df)
    assert counts.tolist() == [1]


def test_calibrate_and_flag_requires_two_votes():
    flagger = ConfidenceFlagger(important_fields=["a"])
    width_ratio_cal = np.linspace(0, 1, 100)
    novelty_cal = np.linspace(0, 1, 100)
    flagger.calibrate(width_ratio_cal, novelty_cal)
    # 90th pct threshold ~0.9, 97th pct (extreme) threshold ~0.97; a value
    # just above the 90th-pct threshold but below the extreme one exercises
    # the "needs a second vote" path without accidentally tripping the
    # single-signal extreme override.
    just_above_90th = 0.92

    # Only one signal above threshold, zero missing fields -> 1 vote -> not flagged
    result = flagger.flag(np.array([just_above_90th]), np.array([0.0]), np.array([0]))
    assert not result[0]

    # Two signals above threshold -> flagged
    result = flagger.flag(np.array([just_above_90th]), np.array([just_above_90th]), np.array([0]))
    assert result[0]


def test_extreme_single_signal_overrides_vote_requirement():
    flagger = ConfidenceFlagger(important_fields=["a"])
    width_ratio_cal = np.linspace(0, 1, 100)
    novelty_cal = np.linspace(0, 1, 100)
    flagger.calibrate(width_ratio_cal, novelty_cal)

    # width ratio far beyond the 97th percentile alone should trip the flag
    result = flagger.flag(np.array([100.0]), np.array([0.0]), np.array([0]))
    assert result[0]
