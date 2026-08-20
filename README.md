# House Price Estimator — Reliable Ranges, Not Just a Number

A house-price estimator for two markets — **Ames, Iowa (USA)** and **Indian
metro cities** — that reports, for every prediction:

1. A **predicted price**
2. A **90% price range**, built with conformal prediction and empirically
   verified (on held-out data) to actually contain the true price about
   90% of the time — not just a guessed-looking band
3. A **SHAP-based explanation** of which features pushed the estimate up
   or down relative to a typical house
4. A **low-confidence flag** for houses that don't resemble the training
   data, or whose range comes out unusually wide, with a recommendation
   to get a professional valuation instead

This is a demonstration project, not a licensed valuation service — every
prediction the app returns says so explicitly.

## Why

Most house-price tools (including Zillow's own "Zestimate", whose 2021
bulk-buying business lost $304M in a single quarter partly because
individually-uncertain estimates were acted on as if they were precise)
report one number and one average accuracy figure. That tells a user
nothing about whether *their specific house* is one the model handles
well. This project reports honest, per-house uncertainty instead.

## Results (see `reports/*_evaluation_report.md` for full tables)

| Market | Point model | Test MAPE | Test log-RMSE | CQR 90% coverage |
|---|---|---|---|---|
| Ames (USD) | XGBoost | 7.7% | 0.111 | 88.7% |
| India (INR) | XGBoost | 51.1% | 0.656 | 89.9% |

The India model is markedly less accurate — reported honestly, not
averaged away. The Indian dataset lists **asking prices, not confirmed
sale prices**, has no transaction date, and ~70% of its amenity fields are
unrecorded. This is a real, stated limitation of the *data*, not a bug —
see `reports/india_evaluation_report.md` for the full breakdown by city
and price bracket.

## Project layout

```
src/
├── config.py                  # paths, seed, per-market registry
├── data/                      # acquire, clean, split
├── features/                  # per-market feature engineering + shared preprocessing
├── models/                    # train/tune, quantile models, persistence
├── uncertainty/                # conformal prediction (MAPIE) + baselines
├── explain/                   # SHAP explanations
├── confidence/                 # low-confidence flag
├── evaluation/                 # metrics, coverage/width tables, report writer
└── pipeline.py                 # end-to-end orchestration
app.py                          # Streamlit app
tests/                          # unit tests
reports/                        # generated evaluation reports
artifacts/{ames,india}/         # persisted model bundles (committed, small)
```

## Setup

```bash
pip install -r requirements.txt
```

### 1. Get the data (Kaggle API token required)

Save a Kaggle API access token to `~/.kaggle/access_token` (see
https://www.kaggle.com/settings → API), then:

```bash
python -m src.data.acquire
```

### 2. Run the full pipeline (clean → train → calibrate → evaluate → persist)

```bash
python -m src.pipeline --market all      # or --market ames / --market india
```

This writes `artifacts/{ames,india}/model_bundle.joblib` (used by the app)
and `reports/{ames,india}_evaluation_report.md`.

### 3. Run the app locally

```bash
streamlit run app.py
```

### 4. Run tests

```bash
pytest tests/
```

## Key design decisions

- **Three-way split** (train / calibration / test) per market — the
  calibration split is touched only to fit conformal predictors, never to
  fit or tune the point models; the test split is touched exactly once.
- **Conformal prediction via MAPIE v1** (`SplitConformalRegressor`,
  `ConformalizedQuantileRegressor`) — not the deprecated `MapieRegressor`.
  Compared against a naive fixed-width baseline and a no-range baseline.
- **XGBoost** is the tuned point-prediction model; MAPIE's CQR whitelist
  doesn't include XGBoost, so the adaptive-width arm uses a
  `HistGradientBoostingRegressor` quantile triplet instead (with
  split-conformal also run on that same model, isolating "does
  adaptivity help" from "does the base model differ").
- **No caste/religion/community-composition features, ever.** The
  deprecated Boston Housing dataset is never used.
- **Every result carries a disclaimer** that this is a model estimate for
  the training data's collection period, not a professional valuation.

## Datasets

- Ames Housing (De Cock, 2011) via
  [kaggle.com/datasets/prevek18/ames-housing](https://www.kaggle.com/datasets/prevek18/ames-housing)
- Housing Prices in Metropolitan Areas of India via
  [kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india](https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india)
