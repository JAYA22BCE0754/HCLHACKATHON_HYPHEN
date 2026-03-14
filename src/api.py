from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
PREDICTION_DECREASE = 0.01


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    model: str
    predictions: List[float]
    n_records: int


app = FastAPI(title="Regression Prediction API", version="1.0.0")

_model = None
_metadata = None


def load_artifacts() -> tuple[Any, Dict[str, Any]]:
    global _model, _metadata

    if _model is None:
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise RuntimeError("Model artifacts not found. Run training first: python -m src.train")

        _model = joblib.load(MODEL_PATH)
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            _metadata = json.load(file)

    return _model, _metadata


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        model, metadata = load_artifacts()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    features = metadata["feature_columns"]
    data = pd.DataFrame(payload.records)

    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing_features}",
        )

    data = data[features]

    try:
        predictions = model.predict(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    # Apply user-requested calibration: decrease each prediction by 0.01.
    adjusted_predictions = [max(0.0, float(value) - PREDICTION_DECREASE) for value in predictions]

    return PredictResponse(
        model=metadata["best_model"],
        predictions=adjusted_predictions,
        n_records=len(adjusted_predictions),
    )
