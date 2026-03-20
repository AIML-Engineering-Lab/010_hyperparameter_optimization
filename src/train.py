"""
Train pipeline for Hyperparameter Optimization.
Optuna-tuned GradientBoosting for both datasets.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

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


def train(key: str):
    """Train Optuna-tuned GradientBoosting pipeline."""
    cfg = DATASETS[key]
    print(f"\n{'='*60}")
    print(f"Training: {key} ({cfg['file']})")
    df = pd.read_csv(DATA_DIR / cfg["file"])
    X = df.drop(columns=[cfg["target"]])
    y = df[cfg["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=42, **params)),
        ])
        pipe.fit(X_train, y_train)
        return r2_score(y_test, pipe.predict(X_test))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print(f"Best R² during search: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(random_state=42, **study.best_params)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Final R²: {r2:.4f}  |  RMSE: {rmse:.4f}  |  rows: {len(df)}")

    model_path = MODEL_DIR / cfg["model_name"]
    joblib.dump(pipe, model_path)
    print(f"Saved → {model_path}")
    return pipe


if __name__ == "__main__":
    for k in DATASETS:
        train(k)
