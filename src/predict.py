"""
Inference for Hyperparameter Optimization.
Load trained model and run predictions on new data.
"""
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

DATASETS = {
    "printer": {
        "file": "3d_printer_quality.csv",
        "target": "print_quality",
        "model_name": "gbr_printer.pkl",
    },
    "pll": {
        "file": "pll_loop_filter.csv",
        "target": "lock_time_us",
        "model_name": "gbr_pll.pkl",
    },
}


def predict(data: pd.DataFrame, model_path: str = None) -> list:
    """Load model and predict on input DataFrame."""
    if model_path is None:
        model_path = str(MODEL_DIR / "gbr_printer.pkl")
    pipe = joblib.load(model_path)
    preds = pipe.predict(data)
    return preds.tolist()


if __name__ == "__main__":
    for key, cfg in DATASETS.items():
        df = pd.read_csv(DATA_DIR / cfg["file"])
        features = df.drop(columns=[cfg["target"]]).head(5)
        preds = predict(features, str(MODEL_DIR / cfg["model_name"]))
        print(f"{key}: {preds}")
