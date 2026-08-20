# Evaluation Report — ames (USD)

## Point-prediction accuracy (best model: xgboost)

|   rmse_log |     mae |    rmse |    mape |
|-----------:|--------:|--------:|--------:|
|   0.111339 | 13226.5 | 20912.7 | 7.66346 |

## CV leaderboard (RMSE, log-price, train-only)

|   linear |   random_forest |   xgboost |
|---------:|----------------:|----------:|
| 0.118744 |        0.135251 |  0.118292 |

## 90% interval coverage & width (target coverage = 90%)

| arm                                                      |   coverage |   avg_width |   median_width |
|:---------------------------------------------------------|-----------:|------------:|---------------:|
| CQR (adaptive, xgboost+HistGBR quantiles)                |   0.887179 |     61963.5 |        50663.6 |
| Split-conformal (xgboost)                                |   0.892308 |     52073.1 |        46066   |
| Split-conformal (HistGBR median, same base model as CQR) |   0.882051 |     52126.4 |        45896.3 |
| Naive fixed-width baseline                               |   0.88547  |     51662.5 |        51662.5 |

## Accuracy by price bracket

| bracket             |   n |      mae |    rmse |     mape |
|:--------------------|----:|---------:|--------:|---------:|
| Q1 (cheapest)       | 147 |  9953.85 | 12997.4 | 10.5452  |
| Q2                  | 148 |  8765.5  | 16875.1 |  6.06071 |
| Q3                  | 144 | 11322.2  | 14788.5 |  6.21363 |
| Q4 (most expensive) | 146 | 22922    | 32831.2 |  7.81668 |

## CQR coverage & width by neighborhood (min n=100)

| segment   |   n | coverage   | avg_width   | note                                      |
|:----------|----:|:-----------|:------------|:------------------------------------------|
| NAmes     |  98 |            |             | insufficient data for a reliable estimate |
| OldTown   |  46 |            |             | insufficient data for a reliable estimate |
| Gilbert   |  43 |            |             | insufficient data for a reliable estimate |
| CollgCr   |  43 |            |             | insufficient data for a reliable estimate |
| Edwards   |  39 |            |             | insufficient data for a reliable estimate |
| Somerst   |  39 |            |             | insufficient data for a reliable estimate |
| NridgHt   |  30 |            |             | insufficient data for a reliable estimate |
| NWAmes    |  28 |            |             | insufficient data for a reliable estimate |
| Mitchel   |  26 |            |             | insufficient data for a reliable estimate |
| Sawyer    |  22 |            |             | insufficient data for a reliable estimate |
| SawyerW   |  19 |            |             | insufficient data for a reliable estimate |
| BrkSide   |  18 |            |             | insufficient data for a reliable estimate |
| NoRidge   |  17 |            |             | insufficient data for a reliable estimate |
| Crawfor   |  17 |            |             | insufficient data for a reliable estimate |
| IDOTRR    |  16 |            |             | insufficient data for a reliable estimate |
| Timber    |  14 |            |             | insufficient data for a reliable estimate |
| StoneBr   |  12 |            |             | insufficient data for a reliable estimate |
| BrDale    |  10 |            |             | insufficient data for a reliable estimate |
| ClearCr   |  10 |            |             | insufficient data for a reliable estimate |
| SWISU     |   9 |            |             | insufficient data for a reliable estimate |
| Blmngtn   |   8 |            |             | insufficient data for a reliable estimate |
| MeadowV   |   6 |            |             | insufficient data for a reliable estimate |
| Veenker   |   6 |            |             | insufficient data for a reliable estimate |
| Blueste   |   3 |            |             | insufficient data for a reliable estimate |
| NPkVill   |   3 |            |             | insufficient data for a reliable estimate |
| Greens    |   1 |            |             | insufficient data for a reliable estimate |
| GrnHill   |   1 |            |             | insufficient data for a reliable estimate |
| Landmrk   |   1 |            |             | insufficient data for a reliable estimate |

## Confidence-flag validation (flagged rows should show worse error)

|   n |     mae |    rmse |     mape |
|----:|--------:|--------:|---------:|
|  38 | 15917.4 | 22408.4 | 13.6592  |
| 547 | 13039.6 | 20804.8 |  7.24693 |
