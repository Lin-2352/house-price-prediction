# House Price Estimator — Reliable Ranges, Not Just a Number

A house-price estimator for **Indian metro cities** (Bangalore, Chennai,
Delhi, Hyderabad, Kolkata, Mumbai) that reports, for every prediction:

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

## Two apps, same backend

| | Primary app | Streamlit app |
|---|---|---|
| **What** | Custom HTML/CSS/JS frontend + FastAPI JSON API | Streamlit's auto-generated widget UI |
| **Where** | Render.com (`web/`, `api/`) | Streamlit Community Cloud (`app.py`) |
| **Why** | Polished, purpose-built UI | Fastest to iterate on, kept live as a fallback |

Both call the exact same trained model/conformal/confidence-flag objects
in `artifacts/india/` — no ML logic is duplicated or diverges between
them. The FastAPI service uses a trimmed `api_bundle.joblib` (see
`src/models/persist.py::save_api_bundle`) that avoids `shap` as a runtime
dependency (XGBoost's native `pred_contribs` produces identical values for
a tree ensemble, without the memory overhead of importing `shap`'s
numba/llvmlite chain) — relevant on Render's 512MB free tier.

## Why

Most house-price tools (including Zillow's own "Zestimate", whose 2021
bulk-buying business lost $304M in a single quarter partly because
individually-uncertain estimates were acted on as if they were precise)
report one number and one average accuracy figure. That tells a user
nothing about whether *their specific house* is one the model handles
well. This project reports honest, per-house uncertainty instead.

## Results (India, test set touched once — see `reports/india_evaluation_report.md` for full tables)

| Point model | Test MAPE | Test log-RMSE | CQR 90% coverage |
|---|---|---|---|
| XGBoost | 51.1% | 0.656 | 89.9% |

This accuracy is markedly weaker than a well-behaved regression problem —
reported honestly, not averaged away. The Indian dataset lists **asking
prices, not confirmed sale prices**, has no transaction date, and ~70% of
its amenity fields are unrecorded. This is a real, stated limitation of
the *data*, not a bug. The app's "Model performance & validation" panel
shows the full breakdown by city and price bracket so any prediction can
be cross-checked against how the model actually performs on similar
houses.

The underlying pipeline code (`src/`) is market-agnostic and originally
also trained an Ames Housing (USA) model as a cross-check that the
approach works on a cleaner dataset (7.7% MAPE there) — that artifact
still exists under `artifacts/ames/` for reference, but the deployed app
is India-only per current scope.

## Project layout

```
src/
├── config.py                  # paths, seed, per-market registry
├── data/                      # acquire, clean, split
├── features/                  # per-market feature engineering + shared preprocessing
├── models/                    # train/tune, quantile models, persistence
├── uncertainty/                # conformal prediction (MAPIE) + baselines
├── explain/
│   ├── feature_names.py       # shap-independent column-name aggregation (shared)
│   └── shap_explainer.py      # SHAP explanations (pipeline/Streamlit only)
├── confidence/                 # low-confidence flag
├── evaluation/                 # metrics, coverage/width tables, report writer
└── pipeline.py                 # end-to-end orchestration
api/
├── main.py                     # FastAPI app: /api/schema, /api/predict, /api/metrics
└── inference.py                # request handling, reuses src/ (no ML logic duplicated)
web/                             # static HTML/CSS/JS frontend served by api/main.py
app.py                          # Streamlit app (kept live as a fallback)
tests/                          # unit tests
reports/                        # generated evaluation reports
artifacts/{ames,india}/         # model_bundle.joblib (full, Streamlit) + api_bundle.joblib (trimmed, FastAPI)
render.yaml                     # Render.com Blueprint for the FastAPI service
requirements-api.txt            # slim deps for the deployed API (no shap/streamlit/plotly/kaggle)
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

### 3. Run an app locally

```bash
streamlit run app.py                              # Streamlit app
uvicorn api.main:app --reload --port 8000          # FastAPI + custom frontend, at localhost:8000
```

### 4. Run tests

```bash
pytest tests/
```

## Deploying the FastAPI app (Render.com)

1. Push the repo to GitHub (`artifacts/india/api_bundle.joblib` must be committed — it's small, ~2.6MB).
2. On Render: **New → Blueprint**, point it at the repo. `render.yaml` defines
   everything (build command, start command, Python version pin) — no manual
   dashboard configuration needed.
3. **Run a single worker.** `render.yaml`'s start command
   (`uvicorn api.main:app --host 0.0.0.0 --port $PORT`) already omits
   `--workers`; do not add one — each additional worker loads its own copy
   of the model bundle into memory, and Render's free tier is 512MB.
4. Free tier sleeps after 15 minutes idle (~1 min cold start on the next
   request) — expected for a demo, not a bug.

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
