"""Tests for Hyperparameter Optimization models."""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_printer_model_exists():
    assert (ROOT / "models" / "gbr_printer.pkl").exists()


def test_pll_model_exists():
    assert (ROOT / "models" / "gbr_pll.pkl").exists()


def test_printer_prediction():
    from predict import predict
    df = pd.read_csv(ROOT / "data" / "3d_printer_quality.csv")
    features = df.drop(columns=["print_quality"]).head(3)
    preds = predict(features, str(ROOT / "models" / "gbr_printer.pkl"))
    assert len(preds) == 3


def test_pll_prediction():
    from predict import predict
    df = pd.read_csv(ROOT / "data" / "pll_loop_filter.csv")
    features = df.drop(columns=["lock_time_us"]).head(3)
    preds = predict(features, str(ROOT / "models" / "gbr_pll.pkl"))
    assert len(preds) == 3


if __name__ == "__main__":
    test_printer_model_exists()
    test_pll_model_exists()
    test_printer_prediction()
    test_pll_prediction()
    print("All 4 tests passed.")
