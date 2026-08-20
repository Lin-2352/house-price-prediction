"""End-to-end orchestration: run every stage for one market and persist
the deployable bundle + evaluation report.

Usage:
    python -m src.pipeline --market ames
    python -m src.pipeline --market india
    python -m src.pipeline --market all
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from src.confidence.flag import ConfidenceFlagger
from src.config import CONFIDENCE_LEVEL, DATA_INTERIM, MIN_SEGMENT_N, get_market
from src.data import clean_ames, clean_india
from src.data.splits import make_splits
from src.evaluation import evaluate as ev
from src.explain.shap_explainer import MarketExplainer
from src.features import ames_features, india_features
from src.features.common import build_form_schema, inverse_log1p, log1p_columns
from src.models import persist, train
from src.models.quantile_models import train_quantile_triplet
from src.uncertainty import baselines
from src.uncertainty.conformal import fit_cqr, fit_split_conformal, predict_interval_currency

MARKET_MODULES = {
    "ames": (clean_ames, ames_features),
    "india": (clean_india, india_features),
}


def run_market(name: str) -> None:
    t0 = time.time()
    print(f"\n{'=' * 70}\nMARKET: {name}\n{'=' * 70}")
    market = get_market(name)
    clean_mod, feat_mod = MARKET_MODULES[name]

    # 1. clean (reuse interim parquet if present, else re-clean from raw)
    interim_path = DATA_INTERIM / f"{name}_clean.parquet"
    if interim_path.exists():
        df = pd.read_parquet(interim_path)
    else:
        df = clean_mod.clean()
        df.to_parquet(interim_path, index=False)

    # 2. derive features (raw units -- house_age/total_area etc. computed
    #    from raw, not-yet-logged square footage/year columns)
    df = feat_mod.derive(df)

    # 3. three-way split on RAW (not yet log-transformed) data, so the
    #    form schema built below stays in human-readable units (real
    #    square feet, not log units). log1p is a stateless per-row
    #    transform, so applying it after splitting each part separately
    #    is mathematically identical to applying it once before splitting.
    splits = make_splits(df, market)
    train_raw, cal_raw, test_raw = splits["train"], splits["calibration"], splits["test"]

    spec = feat_mod.feature_spec()
    form_schema = build_form_schema(train_raw.drop(columns=[market.price_col]), spec)

    # 4. log-transform price + primary area column (both right-skewed) --
    #    applied per split, after the split, for the reason above.
    train_df = log1p_columns(train_raw, list(market.log_cols))
    cal_df = log1p_columns(cal_raw, list(market.log_cols))
    test_df = log1p_columns(test_raw, list(market.log_cols))

    y_train_log = train_df[market.price_col]
    y_cal_log = cal_df[market.price_col]
    y_test_log = test_df[market.price_col]
    X_train = train_df.drop(columns=[market.price_col])
    X_cal = cal_df.drop(columns=[market.price_col])
    X_test = test_df.drop(columns=[market.price_col])

    # 5. train + tune the three point models on train only
    results = train.train_all(spec, X_train, y_train_log)
    best_name = results["best_name"]
    best_model = results[best_name]
    print(f"Selected point model: {best_name}")

    # 6. quantile triplet (HistGBR) for CQR, trained on train only
    print("Training quantile triplet for CQR...")
    triplet = train_quantile_triplet(spec, X_train, y_train_log)
    median_model = triplet[2]

    # 7. conformal calibration on the calibration split (never used for fitting)
    print("Calibrating conformal predictors...")
    split_conf_best = fit_split_conformal(best_model, X_cal, y_cal_log)
    split_conf_histgbr = fit_split_conformal(median_model, X_cal, y_cal_log)
    cqr = fit_cqr(triplet, X_cal, y_cal_log)

    # 8. naive fixed-width baseline, calibrated on the calibration split
    point_cal_currency = inverse_log1p(best_model.predict(X_cal))
    y_cal_currency = inverse_log1p(y_cal_log)
    fixed_half_width = baselines.calibrate_fixed_width(y_cal_currency, point_cal_currency, CONFIDENCE_LEVEL)
    print(f"Naive fixed-width baseline half-width: {market.currency_symbol}{fixed_half_width:,.0f}")

    # 9. confidence flagger: fit novelty model on train, calibrate thresholds on calibration
    print("Fitting confidence flag signals...")
    important_fields = feat_mod.IMPORTANT_FIELDS
    flagger = ConfidenceFlagger(important_fields=important_fields)
    X_train_transformed = best_model.named_steps["prep"].transform(X_train)
    flagger.fit_novelty(X_train_transformed)

    X_cal_transformed = best_model.named_steps["prep"].transform(X_cal)
    novelty_cal = flagger.novelty_score(X_cal_transformed)
    _, cqr_lower_cal, cqr_upper_cal = predict_interval_currency(cqr, X_cal)
    width_ratio_cal = (cqr_upper_cal - cqr_lower_cal) / np.clip(point_cal_currency, 1, None)
    flagger.calibrate(width_ratio_cal, novelty_cal)

    # 10. evaluation on test (touched once)
    print("Evaluating on test split...")
    y_test_currency = inverse_log1p(y_test_log)

    point_pred_log_best = best_model.predict(X_test)
    point_pred_currency_best = inverse_log1p(point_pred_log_best)

    _, cqr_lo, cqr_hi = predict_interval_currency(cqr, X_test)
    _, sc_best_lo, sc_best_hi = predict_interval_currency(split_conf_best, X_test)
    _, sc_hist_lo, sc_hist_hi = predict_interval_currency(split_conf_histgbr, X_test)
    fixed_lo, fixed_hi = baselines.predict_fixed_width(point_pred_currency_best, fixed_half_width)

    point_accuracy = ev.point_accuracy(y_test_log, point_pred_log_best, y_test_currency, point_pred_currency_best)

    coverage_table = ev.coverage_width_table(y_test_currency, {
        f"CQR (adaptive, {best_name}+HistGBR quantiles)": (cqr_lo, cqr_hi),
        f"Split-conformal ({best_name})": (sc_best_lo, sc_best_hi),
        "Split-conformal (HistGBR median, same base model as CQR)": (sc_hist_lo, sc_hist_hi),
        "Naive fixed-width baseline": (fixed_lo, fixed_hi),
    })

    bracket_table = ev.bracket_accuracy(y_test_currency, pd.Series(point_pred_currency_best, index=y_test_currency.index))

    segment_series = test_df[market.segment_col] if market.segment_col in test_df.columns else X_test[market.segment_col]
    segment_table_cqr = ev.segmented_coverage_width(y_test_currency, cqr_lo, cqr_hi, segment_series, MIN_SEGMENT_N)

    X_test_transformed = best_model.named_steps["prep"].transform(X_test)
    novelty_test = flagger.novelty_score(X_test_transformed)
    width_ratio_test = (cqr_hi - cqr_lo) / np.clip(point_pred_currency_best, 1, None)
    missing_test = flagger.missing_field_count(X_test)
    flagged_test = flagger.flag(width_ratio_test, novelty_test, missing_test)
    flag_validation = ev.validate_confidence_flag(y_test_currency, pd.Series(point_pred_currency_best, index=y_test_currency.index), flagged_test)

    print(f"Point accuracy: {point_accuracy}")
    print(coverage_table.to_string(index=False))
    print(f"Confidence flag validation: {flag_validation}")

    # 11. SHAP explainer for the best model
    print("Building SHAP explainer...")
    explainer = MarketExplainer(best_model, spec, X_train)

    # 12. persist deployable bundle
    bundle = {
        "market": name,
        "spec": spec,
        "point_model": best_model,
        "point_model_name": best_name,
        "quantile_triplet": triplet,
        "cqr": cqr,
        "split_conformal": split_conf_best,
        "fixed_half_width": fixed_half_width,
        "confidence_flagger": flagger,
        "explainer": explainer,
        "important_fields": important_fields,
        "form_schema": form_schema,
        "currency_symbol": market.currency_symbol,
        "currency_code": market.currency_code,
        "price_col": market.price_col,
        "segment_col": market.segment_col,
        "log_feature_cols": [c for c in market.log_cols if c != market.price_col],
        "test_rmse_log": point_accuracy["rmse_log"],
        "test_mape": point_accuracy["mape"],
        "cv_rmse_log": results["cv_rmse_log"],
    }
    persist.save_bundle(market.artifact_dir, bundle)
    persist.save_leaderboard(market.artifact_dir, {
        "cv_rmse_log": results["cv_rmse_log"],
        "models": {k: results[k] for k in ("linear", "random_forest", "xgboost")},
    })
    persist.save_json(market.artifact_dir, "metrics.json", {
        "point_accuracy": point_accuracy,
        "coverage_table": coverage_table.to_dict(orient="records"),
        "bracket_table": bracket_table.to_dict(orient="records"),
        "segment_table": segment_table_cqr.to_dict(orient="records"),
        "confidence_flag_validation": flag_validation,
        "cv_rmse_log": results["cv_rmse_log"],
    })

    ev.write_report(market, {
        "Point-prediction accuracy (best model: " + best_name + ")": pd.DataFrame([point_accuracy]),
        "CV leaderboard (RMSE, log-price, train-only)": pd.DataFrame([results["cv_rmse_log"]]),
        f"90% interval coverage & width (target coverage = {CONFIDENCE_LEVEL:.0%})": coverage_table,
        "Accuracy by price bracket": bracket_table,
        f"CQR coverage & width by {market.segment_col} (min n={MIN_SEGMENT_N})": segment_table_cqr,
        "Confidence-flag validation (flagged rows should show worse error)": pd.DataFrame(flag_validation).T,
    }, market.artifact_dir.parent.parent / "reports" / f"{name}_evaluation_report.md")

    print(f"\nDone with {name} in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["ames", "india", "all"], default="all")
    args = parser.parse_args()
    markets = ["ames", "india"] if args.market == "all" else [args.market]
    for m in markets:
        run_market(m)


if __name__ == "__main__":
    main()
