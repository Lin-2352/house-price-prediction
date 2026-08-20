"""Save/load one versioned joblib bundle per market.

Only the artifacts the deployed app actually needs are kept in the bundle
that ships to Streamlit Cloud-sized memory: the chosen point model, both
conformal objects, the quantile triplet (needed to reproduce CQR at
inference), the SHAP explainer's small background sample, the confidence
flagger, and a metadata dict. The full CV leaderboard (all three candidate
models) is saved separately to artifacts/<market>/leaderboard.joblib for
reference/reporting but is NOT loaded by the app.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib

BUNDLE_VERSION = 1


def save_bundle(market_dir: Path, bundle: dict) -> Path:
    bundle = dict(bundle)
    bundle["_version"] = BUNDLE_VERSION
    path = market_dir / "model_bundle.joblib"
    joblib.dump(bundle, path, compress=3)
    print(f"Saved bundle to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_bundle(market_dir: Path) -> dict:
    path = market_dir / "model_bundle.joblib"
    bundle = joblib.load(path)
    if bundle.get("_version") != BUNDLE_VERSION:
        print(f"Warning: bundle version mismatch ({bundle.get('_version')} != {BUNDLE_VERSION})")
    return bundle


def save_leaderboard(market_dir: Path, leaderboard: dict) -> None:
    path = market_dir / "leaderboard.joblib"
    joblib.dump(leaderboard, path, compress=3)


def save_json(market_dir: Path, name: str, obj) -> None:
    (market_dir / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
