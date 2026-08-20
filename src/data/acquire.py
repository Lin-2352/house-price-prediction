"""Download the two source datasets from Kaggle.

Requires a Kaggle API access token at ~/.kaggle/access_token (the modern
token-based auth flow) or a legacy ~/.kaggle/kaggle.json username+key file.
This script only touches data/raw/ — nothing here is committed to git.
"""
from __future__ import annotations

import subprocess
import sys

from src.config import DATA_RAW

AMES_DATASET = "prevek18/ames-housing"
INDIA_DATASET = "ruchi798/housing-prices-in-metropolitan-areas-of-india"


def _download(dataset: str, subdir: str) -> None:
    dest = DATA_RAW / subdir
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    _download(AMES_DATASET, "ames")
    _download(INDIA_DATASET, "india")
    print("Done. Raw files are under data/raw/{ames,india}/")


if __name__ == "__main__":
    sys.exit(main())
