"""Crop yield prediction using the trained model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_CANDIDATES = (
    "random_forest_weather.joblib",
    "random_forest.joblib",
    "best_model.pkl",
    "xgboost_weather.joblib",
    "xgboost.joblib",
)


def _resolve_feature_names(model, features=None):
    if features is not None:
        return list(features)

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    transformer = getattr(model, "named_steps", {}).get("preprocessor")
    if transformer is not None and hasattr(transformer, "get_feature_names_out"):
        try:
            return list(transformer.get_feature_names_out())
        except (AttributeError, ValueError, TypeError):
            pass

    return None


def load_artifacts(model_dir=MODEL_DIR):
    """Load the primary model artifact and its declared feature list."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_path = None
    for candidate in MODEL_CANDIDATES:
        path = model_dir / candidate
        if path.exists():
            model_path = path
            break

    if model_path is None:
        joblib_candidates = sorted(model_dir.glob("*.joblib")) + sorted(model_dir.glob("*.pkl"))
        if not joblib_candidates:
            raise FileNotFoundError(
                "No trained model artifacts found in the models directory. "
                "Run src/train.py or src/train_weather.py first."
            )
        model_path = joblib_candidates[0]

    model = joblib.load(model_path)
    features = _resolve_feature_names(model)
    if features is None:
        features_path = model_dir / "features.pkl"
        if features_path.exists():
            features = joblib.load(features_path)

    return model, features


def predict_yield(input_data, model=None, features=None):
    """
    Predict yield in the target's original units (normally tons/ha).

    input_data may be a dictionary, Series, or one-row DataFrame.
    Missing model features are filled with NaN so the trained
    pipeline can apply its fitted imputers.
    """
    if model is None:
        model, features = load_artifacts()
    elif features is None:
        features = _resolve_feature_names(model)

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

    # Match the case-insensitive column naming used by the training scripts.
    aliases = {
        "state": "State",
        "state_name": "State",
        "statename": "State",
        "district": "District",
        "district_name": "District",
        "districtname": "District",
        "crop": "Crop",
        "season": "Season",
        "year": "Year",
        "cropyear": "Year",
        "crop_year": "Year",
        "rainfall": "Rainfall",
        "rain": "Rainfall",
        "avgtemperature": "AvgTemperature",
        "avg_temperature": "AvgTemperature",
        "avghumidity": "AvgHumidity",
        "avg_humidity": "AvgHumidity",
        "temperature": "temperature",
        "ph": "pH",
        "n": "N",
        "p": "P",
        "k": "K",
        "mean_temperature": "Mean_Temperature",
        "avg_max_temp": "Avg_Max_Temp",
        "avgmaxtemp": "Avg_Max_Temp",
        "avg_min_temp": "Avg_Min_Temp",
        "avgmintemp": "Avg_Min_Temp",
        "total_precipitation_mm": "Total_Precipitation_mm",
        "totalprecipitationmm": "Total_Precipitation_mm",
        "max_wind_speed": "Max_Wind_Speed",
        "maxwindspeed": "Max_Wind_Speed",
    }

    rename = {}
    for column in row.columns:
        key = column.casefold().replace(" ", "").replace("_", "")
        if key in aliases:
            rename[column] = aliases[key]
    row = row.rename(columns=rename)

    if "AvgTemperature" not in row.columns and {"temp_max_c", "temp_min_c"}.issubset(row.columns):
        row["AvgTemperature"] = (
            pd.to_numeric(row["temp_max_c"], errors="coerce")
            + pd.to_numeric(row["temp_min_c"], errors="coerce")
        ) / 2

    if "Rainfall" not in row.columns:
        if "rain_mm" in row.columns:
            row["Rainfall"] = row["rain_mm"]
        elif "precipitation_mm" in row.columns:
            row["Rainfall"] = row["precipitation_mm"]

    if features is None:
        raise ValueError("Could not determine the model's feature names from the artifact.")

    for column in list(row.columns):
        if column in {
            "Year",
            "Area",
            "Production",
            "Yield",
            "Rainfall",
            "AvgTemperature",
            "AvgHumidity",
            "N",
            "P",
            "K",
            "pH",
            "Mean_Temperature",
            "Avg_Max_Temp",
            "Avg_Min_Temp",
            "Total_Precipitation_mm",
            "Max_Wind_Speed",
        }:
            row[column] = pd.to_numeric(row[column], errors="coerce")

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
