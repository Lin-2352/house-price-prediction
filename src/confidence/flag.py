"""Low-confidence flag: combines three signals, each thresholded at its
90th percentile measured on the calibration split (never train or test),
and trips when at least CONFIDENCE_SIGNAL_VOTES_REQUIRED of them fire.

Signals:
  1. range_width_ratio -- (upper - lower) / predicted price. A wide range
     already means the model itself is unsure; flagging on top of that
     tells the user *why* a range is wide instead of leaving them to guess.
  2. novelty -- an IsolationForest fit on the training split's
     preprocessed features (never calibration/test), scored at inference.
     High novelty means the house doesn't resemble anything the model was
     trained on, so both the point prediction and its range are on shakier
     ground than the calibration guarantee accounts for.
  3. missing_important_fields -- count of high-importance raw inputs the
     user left blank (matters most in the live app; on already-complete
     dataset rows this is usually 0).

Whether a flagged prediction is actually worse must be verified on the
test split after fitting (see evaluation/evaluate.py) -- that's the
evidence the flag is a real signal and not a cosmetic feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import CONFIDENCE_SIGNAL_VOTES_REQUIRED, RANDOM_SEED


@dataclass
class ConfidenceThresholds:
    width_ratio: float
    novelty: float
    missing_fields: float = 0.5  # any missing important field is notable
    # "Extreme" single-signal override thresholds (97th percentile on
    # calibration). Added after discovering empirically (India market)
    # that range-width and novelty can be *anti-correlated* -- a house
    # whose quantile-regression interval is unusually wide isn't always
    # the same house the novelty detector considers unusual, so requiring
    # 2-of-3 agreement can under-flag when only two signals are practically
    # active (missing-fields rarely fires on complete dataset rows). A
    # single signal in its own extreme tail is still worth a warning on
    # its own, regardless of what the other signals say.
    width_ratio_extreme: float = float("inf")
    novelty_extreme: float = float("inf")


@dataclass
class ConfidenceFlagger:
    important_fields: list[str]
    novelty_model: IsolationForest = field(default=None, repr=False)
    thresholds: ConfidenceThresholds | None = None

    def fit_novelty(self, X_train_transformed: np.ndarray) -> None:
        self.novelty_model = IsolationForest(
            n_estimators=300, contamination="auto", random_state=RANDOM_SEED, n_jobs=-1
        )
        self.novelty_model.fit(X_train_transformed)

    def novelty_score(self, X_transformed: np.ndarray) -> np.ndarray:
        # Higher = more novel/unusual (score_samples: higher = more normal)
        return -self.novelty_model.score_samples(X_transformed)

    def missing_field_count(self, x_raw: pd.DataFrame) -> np.ndarray:
        present = x_raw[self.important_fields].notna()
        return (len(self.important_fields) - present.sum(axis=1)).to_numpy()

    def calibrate(self, width_ratio_cal: np.ndarray, novelty_cal: np.ndarray) -> None:
        self.thresholds = ConfidenceThresholds(
            width_ratio=float(np.quantile(width_ratio_cal, 0.90)),
            novelty=float(np.quantile(novelty_cal, 0.90)),
            width_ratio_extreme=float(np.quantile(width_ratio_cal, 0.97)),
            novelty_extreme=float(np.quantile(novelty_cal, 0.97)),
        )

    def flag(self, width_ratio: np.ndarray, novelty: np.ndarray, missing_count: np.ndarray) -> np.ndarray:
        assert self.thresholds is not None, "call calibrate() first"
        votes = (
            (width_ratio > self.thresholds.width_ratio).astype(int)
            + (novelty > self.thresholds.novelty).astype(int)
            + (missing_count >= self.thresholds.missing_fields).astype(int)
        )
        extreme = (
            (width_ratio > self.thresholds.width_ratio_extreme)
            | (novelty > self.thresholds.novelty_extreme)
        )
        return (votes >= CONFIDENCE_SIGNAL_VOTES_REQUIRED) | extreme

    def explain_flag(self, width_ratio: float, novelty: float, missing_count: int) -> list[str]:
        assert self.thresholds is not None
        reasons = []
        if width_ratio > self.thresholds.width_ratio_extreme:
            reasons.append("the predicted price range is extremely wide for this price level")
        elif width_ratio > self.thresholds.width_ratio:
            reasons.append("the predicted price range is unusually wide for this price level")
        if novelty > self.thresholds.novelty_extreme:
            reasons.append("this house's features are extremely unlike the training data")
        elif novelty > self.thresholds.novelty:
            reasons.append("this house's features are unlike most houses the model was trained on")
        if missing_count >= self.thresholds.missing_fields:
            reasons.append("one or more important fields were left blank")
        return reasons
