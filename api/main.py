"""FastAPI service: JSON API + static frontend for the India house-price
estimator. Run with a single worker (see render.yaml / README) -- multiple
workers would each load a separate copy of the model bundle into memory.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.inference import Model

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

_model: Model | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = Model()  # loaded once at startup, reused across requests
    yield


app = FastAPI(title="House Price Estimator API", lifespan=lifespan)


class PredictRequest(BaseModel):
    values: dict
    range_method: str = "cqr"  # "cqr" | "split_conformal"


@app.get("/api/schema")
def get_schema():
    return _model.schema()


@app.get("/api/metrics")
def get_metrics():
    return _model.metrics_payload()


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.range_method not in ("cqr", "split_conformal"):
        raise HTTPException(422, "range_method must be 'cqr' or 'split_conformal'")
    try:
        return _model.predict(req.values, req.range_method)
    except Exception as e:  # noqa: BLE001 -- surface a clean 400 instead of a 500 traceback
        raise HTTPException(400, f"Could not compute a prediction from the given inputs: {e}") from e


# Mounted last so /api/* routes above take priority over the catch-all
# static file server.
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
