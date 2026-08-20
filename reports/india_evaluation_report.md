# Evaluation Report — india (INR)

## Point-prediction accuracy (best model: xgboost)

|   rmse_log |         mae |       rmse |    mape |
|-----------:|------------:|-----------:|--------:|
|   0.655933 | 6.02146e+06 | 2.0161e+07 | 51.0644 |

## CV leaderboard (RMSE, log-price, train-only)

|   linear |   random_forest |   xgboost |
|---------:|----------------:|----------:|
| 0.681564 |        0.654158 |  0.649413 |

## 90% interval coverage & width (target coverage = 90%)

| arm                                                      |   coverage |   avg_width |   median_width |
|:---------------------------------------------------------|-----------:|------------:|---------------:|
| CQR (adaptive, xgboost+HistGBR quantiles)                |   0.899262 | 2.3488e+07  |    2.01983e+07 |
| Split-conformal (xgboost)                                |   0.905269 | 2.19374e+07 |    1.70394e+07 |
| Split-conformal (HistGBR median, same base model as CQR) |   0.903381 | 2.09566e+07 |    1.6351e+07  |
| Naive fixed-width baseline                               |   0.900635 | 2.18437e+07 |    2.18437e+07 |

## Accuracy by price bracket

| bracket             |    n |         mae |        rmse |     mape |
|:--------------------|-----:|------------:|------------:|---------:|
| Q1 (cheapest)       | 1457 | 2.89613e+06 | 4.06819e+06 | 100.459  |
| Q2                  | 1467 | 1.80616e+06 | 6.43146e+06 |  34.8028 |
| Q3                  | 1476 | 2.33099e+06 | 3.11933e+06 |  25.294  |
| Q4 (most expensive) | 1427 | 1.73632e+07 | 3.98783e+07 |  44.0038 |

## CQR coverage & width by city (min n=100)

| segment   |    n |   coverage |   avg_width |   median_width | note   |
|:----------|-----:|-----------:|------------:|---------------:|:-------|
| Mumbai    | 1331 |   0.89707  | 3.32864e+07 |    3.16086e+07 |        |
| Kolkata   | 1260 |   0.892063 | 1.96708e+07 |    1.90177e+07 |        |
| Bangalore | 1107 |   0.918699 | 2.01912e+07 |    2.03357e+07 |        |
| Chennai   |  892 |   0.902466 | 1.56072e+07 |    1.61497e+07 |        |
| Delhi     |  842 |   0.889549 | 3.30807e+07 |    1.59787e+07 |        |
| Hyderabad |  395 |   0.888608 | 9.2356e+06  |    5.28451e+06 |        |

## Confidence-flag validation (flagged rows should show worse error)

|    n |         mae |        rmse |    mape |
|-----:|------------:|------------:|--------:|
|  310 | 8.36386e+06 | 2.53557e+07 | 62.1408 |
| 5517 | 5.88984e+06 | 1.98288e+07 | 50.4421 |
