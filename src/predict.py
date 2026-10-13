"""Crop yield prediction using the trained model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def load_artifacts(model_dir=MODEL_DIR):
    """Load the model and exact feature list saved by train.py."""
    model_dir = Path(model_dir)
    model_path = model_dir / "best_model.pkl"
    features_path = model_dir / "features.pkl"

    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run src/train.py first."
        )

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features


def predict_yield(input_data, model=None, features=None):
    """
    Predict yield in the target's original units (normally tons/ha).

    input_data may be a dictionary, Series, or one-row DataFrame.
    Missing model features are filled with NaN so the trained
    pipeline can apply its fitted imputers.
    """
    if model is None or features is None:
        model, features = load_artifacts()

    if isinstance(input_data, dict):
        row = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        row = input_data.to_frame().T
    elif isinstance(input_data, pd.DataFrame):
        if len(input_data) != 1:
            raise ValueError("Pass exactly one prediction row.")
        row = input_data.copy()
    else:
        raise TypeError(
            "input_data must be a dictionary, Series, or DataFrame."
        )

    row.columns = row.columns.astype(str).str.strip()

    # Match the case-insensitive column naming used by train.py.
    aliases = {
        "state": "State",
        "district": "District",
        "districtname": "District",
        "statename": "State",
        "crop": "Crop",
        "season": "Season",
        "rainfall": "Rainfall",
        "avgtemperature": "AvgTemperature",
        "avghumidity": "AvgHumidity",
        "ph": "pH",
        "n": "N",
        "p": "P",
        "k": "K",
    }

    rename = {}
    for column in row.columns:
        key = column.casefold().replace(" ", "").replace("_", "")
        if key in aliases:
            rename[column] = aliases[key]
    row = row.rename(columns=rename)

    # Support raw daily-weather column names used by the trainer.
    if (
        "AvgTemperature" not in row.columns
        and {"temp_max_c", "temp_min_c"}.issubset(row.columns)
    ):
        row["AvgTemperature"] = (
            pd.to_numeric(row["temp_max_c"], errors="coerce")
            + pd.to_numeric(row["temp_min_c"], errors="coerce")
        ) / 2

    if "Rainfall" not in row.columns:
        if "rain_mm" in row.columns:
            row["Rainfall"] = row["rain_mm"]
        elif "precipitation_mm" in row.columns:
            row["Rainfall"] = row["precipitation_mm"]

    # The feature list is authoritative: same columns, same order.
    row = row.reindex(columns=features)

    prediction = float(np.asarray(model.predict(row)).reshape(-1)[0])
    prediction = max(0.0, prediction)

    return {
        "predicted_yield": round(prediction, 3),
        "unit": "tons/hectare (if that is the unit of training Yield)",
    }


def predict_from_notebook(df, row_index, model=None, features=None):
    """Predict directly from a row in a notebook DataFrame."""
    if model is None or features is None:
        model, features = load_artifacts()

    if row_index not in df.index:
        raise IndexError(f"Row index {row_index} is not in the DataFrame.")

    return predict_yield(
        df.loc[row_index],
        model=model,
        features=features,
    )


if __name__ == "__main__":
    example = {
        "Crop": "Wheat",
        "District": "Example District",
        "Season": "Rabi",
        "Rainfall": 450,
        "AvgTemperature": 24,
        "AvgHumidity": 60,
        "N": 90,
        "P": 40,
        "K": 40,
        "pH": 6.5,
    }

    print(predict_yield(example))