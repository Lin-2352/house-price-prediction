"""Central configuration: paths, random seed, and the per-market registry.

Every pipeline stage and the Streamlit app reads market-specific settings
(currency, which columns to log-transform, artifact locations, ...) from
MARKETS below instead of hardcoding "ames" vs "india" branches everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

for _p in (DATA_INTERIM, DATA_PROCESSED, ARTIFACTS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# Three-way split fractions (must sum to 1.0)
TRAIN_FRAC = 0.60
CAL_FRAC = 0.20
TEST_FRAC = 0.20

# Conformal prediction confidence level
CONFIDENCE_LEVEL = 0.90

# Minimum test-set rows required before reporting a per-segment (city /
# neighborhood) breakdown number; below this we report "insufficient data".
MIN_SEGMENT_N = 100

# Confidence-flag: trip if at least this many of the 3 signals exceed
# their calibration-derived (90th percentile) threshold.
CONFIDENCE_SIGNAL_VOTES_REQUIRED = 2


@dataclass(frozen=True)
class MarketConfig:
    name: str                      # "ames" | "india"
    currency_symbol: str           # "$" | "₹"
    currency_code: str             # "USD" | "INR"
    price_col: str                 # target column name after cleaning
    log_cols: tuple[str, ...]      # columns to log1p-transform (price + area)
    locality_col: str              # column used for target encoding (neighborhood / city+location)
    segment_col: str               # column used for per-segment evaluation breakdown
    artifact_dir: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "artifact_dir", ARTIFACTS / self.name)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


MARKETS: dict[str, MarketConfig] = {
    "ames": MarketConfig(
        name="ames",
        currency_symbol="$",
        currency_code="USD",
        price_col="sale_price",
        log_cols=("sale_price", "gr_liv_area"),
        locality_col="neighborhood",
        segment_col="neighborhood",
    ),
    "india": MarketConfig(
        name="india",
        currency_symbol="₹",
        currency_code="INR",
        price_col="price",
        log_cols=("price", "area"),
        locality_col="location",
        segment_col="city",
    ),
}


def get_market(name: str) -> MarketConfig:
    try:
        return MARKETS[name]
    except KeyError as e:
        raise ValueError(f"Unknown market {name!r}; choose from {list(MARKETS)}") from e
