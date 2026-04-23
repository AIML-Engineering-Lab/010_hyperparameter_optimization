"""
FastAPI serving endpoint for Hyperparameter Optimization.
POST features -> model prediction.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Hyperparameter Optimization API", version="1.0.0")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "gbr_printer.pkl"
_model = None


class PredictionInput(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    prediction: float
    model: str = "GBR-Optuna (3D Printer)"


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "healthy", "model": "GBR-Optuna"}


@app.get("/info")
def info_endpoint():
    return {
        "project": "010_hyperparameter_optimization",
        "description": "Optuna-tuned GradientBoosting",
        "task": "regression",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    try:
        model = get_model()
        df = pd.DataFrame([input_data.features])
        pred = model.predict(df)[0]
        return PredictionResponse(prediction=float(pred))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
