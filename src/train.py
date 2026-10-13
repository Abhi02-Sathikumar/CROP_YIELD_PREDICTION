import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# XGBoost is optional
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "training_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"

TARGET = "Yield"
TEST_FRACTION = 0.20
RANDOM_STATE = 42

# Yield is calculated as Production / Area.
# Exclude these columns to prevent target leakage
# and avoid using the target's source value directly.
EXCLUDED_FEATURES = {
    "Yield",
    "Production",
    "Area",
}


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

def load_data(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {path}\n"
            "Run the notebook exports and then "
            "python src/data_preprocessing.py first."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if "Year" not in df.columns:
        raise ValueError(
            "Year is required for chronological evaluation."
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )

    df["Year"] = pd.to_numeric(
        df["Year"], errors="coerce"
    )

    df = df.dropna(subset=[TARGET, "Year"]).copy()

    # Reject invalid year keys rather than rounding
    # potentially incorrect decimal-year observations.
    df = df[df["Year"] % 1 == 0].copy()
    df["Year"] = df["Year"].astype(int)

    # Yield must be nonnegative.
    df = df[df[TARGET] >= 0].copy()

    if df.empty:
        raise ValueError(
            "No valid training rows remain after cleaning."
        )

    print("Loaded dataset:", df.shape)

    return df


# --------------------------------------------------
# 2. STANDARDIZE COLUMN NAMES
# --------------------------------------------------

def standardize_features(df):
    """
    Normalize known column aliases while preserving the
    original weather measurement names.
    """

    # Keys are normalized by removing spaces and underscores.
    aliases = {
        "state": "State",
        "statename": "State",
        "district": "District",
        "districtname": "District",
        "crop": "Crop",
        "season": "Season",
        "year": "Year",
        "yield": "Yield",
        "production": "Production",
        "area": "Area",
        "ph": "pH",
        "n": "N",
        "p": "P",
        "k": "K",
        "rainfall": "Rainfall",
        "avgtemperature": "AvgTemperature",
        "avgtemperaturec": "AvgTemperature",
        "avghumidity": "AvgHumidity",
    }

    rename_map = {}

    for col in df.columns:
        key = (
            col.strip()
            .casefold()
            .replace(" ", "")
            .replace("_", "")
        )

        if key in aliases:
            rename_map[col] = aliases[key]

    df = df.rename(columns=rename_map)

    # Normalize geographic and categorical fields.
    for col in ["State", "District", "Crop", "Season"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    # Create average temperature from maximum/minimum,
    # but only if both columns exist.
    if (
        "AvgTemperature" not in df.columns
        and "temp_max_c" in df.columns
        and "temp_min_c" in df.columns
    ):
        df["AvgTemperature"] = (
            pd.to_numeric(df["temp_max_c"], errors="coerce")
            + pd.to_numeric(df["temp_min_c"], errors="coerce")
        ) / 2

    # Map weather measurements to a standard rainfall name.
    # Prefer rain_mm if available.
    if "Rainfall" not in df.columns:
        if "rain_mm" in df.columns:
            df["Rainfall"] = pd.to_numeric(
                df["rain_mm"], errors="coerce"
            )
        elif "precipitation_mm" in df.columns:
            df["Rainfall"] = pd.to_numeric(
                df["precipitation_mm"], errors="coerce"
            )

    return df


# --------------------------------------------------
# 3. SELECT MODEL FEATURES
# --------------------------------------------------

def select_features(df):
    """
    Select available predictors for yield prediction.

    Year is retained for splitting but excluded from X.
    Production, Area, and Yield are excluded from X.
    """

    candidate_features = [
        # Crop and location
        "Crop",
        "State",
        "District",
        "Season",

        # Weather
        "Rainfall",
        "AvgTemperature",
        "AvgHumidity",
        "temp_max_c",
        "temp_min_c",
        "temp_range_c",
        "apparent_temp_max_c",
        "apparent_temp_min_c",
        "precipitation_mm",
        "rain_mm",
        "precipitation_hours",
        "sunshine_duration_sec",
        "daylight_duration_sec",
        "wind_speed_max_kmh",
        "wind_gusts_max_kmh",
        "wind_direction_dominant_deg",
        "solar_radiation_mj_m2",
        "reference_evapotranspiration_mm",

        # Soil features, only if genuinely present
        # in the production dataset.
        "N",
        "P",
        "K",
        "pH",
    ]

    features = [
        col for col in candidate_features
        if col in df.columns
        and col not in EXCLUDED_FEATURES
        and col != "Year"
    ]

    if not features:
        raise ValueError(
            "No supported features were found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    categorical_features = [
        col for col in [
            "Crop", "State", "District", "Season"
        ]
        if col in features
    ]

    numeric_features = [
        col for col in features
        if col not in categorical_features
    ]

    # Convert feature types before checking for usable values.
    for col in numeric_features:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    for col in categorical_features:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    all_missing = [
        col for col in features
        if df[col].isna().all()
    ]

    if all_missing:
        warnings.warn(
            "Dropping entirely missing features: "
            + ", ".join(all_missing)
        )
        features = [
            col for col in features
            if col not in all_missing
        ]

    if not features:
        raise ValueError(
            "No usable features remain. "
            "Check the preprocessing output and weather merge."
        )

    # Keep only rows with at least one observed predictor.
    df = df.dropna(
        subset=features,
        how="all"
    ).copy()

    if df.empty:
        raise ValueError(
            "No rows remain with usable model features."
        )

    print("\nSelected features:")
    for col in features:
        print(
            f"  {col}: "
            f"{df[col].notna().sum()} observed values"
        )

    return df, features


# --------------------------------------------------
# 4. CHRONOLOGICAL TRAIN/TEST SPLIT
# --------------------------------------------------

def chronological_split(df):
    """
    Reserve the latest distinct years for testing.
    """

    years = sorted(df["Year"].unique())

    if len(years) < 2:
        raise ValueError(
            "At least two distinct years are required. "
            f"Found: {years}"
        )

    test_year_count = max(
        1,
        int(np.ceil(len(years) * TEST_FRACTION))
    )

    test_year_count = min(
        test_year_count,
        len(years) - 1
    )

    test_years = years[-test_year_count:]
    first_test_year = min(test_years)

    train_df = df[
        df["Year"] < first_test_year
    ].copy()

    test_df = df[
        df["Year"] >= first_test_year
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Chronological split produced an empty dataset."
        )

    print("\nChronological split")
    print("-------------------")
    print(
        f"Training years: {train_df['Year'].min()} - "
        f"{train_df['Year'].max()}"
    )
    print(
        f"Testing years: {test_df['Year'].min()} - "
        f"{test_df['Year'].max()}"
    )
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))

    return train_df, test_df


# --------------------------------------------------
# 5. BUILD PREPROCESSING PIPELINE
# --------------------------------------------------

def build_preprocessor(features):
    categorical_features = [
        col for col in [
            "Crop", "State", "District", "Season"
        ]
        if col in features
    ]

    numeric_features = [
        col for col in features
        if col not in categorical_features
    ]

    transformers = []

    if categorical_features:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if numeric_features:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
        ])

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# --------------------------------------------------
# 6. DEFINE MODELS
# --------------------------------------------------

def get_models():
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        print(
            "\nXGBoost is not installed. "
            "Using Linear Regression and Random Forest."
        )

    return models


# --------------------------------------------------
# 7. TRAIN AND EVALUATE
# --------------------------------------------------

def train_models(train_df, test_df, features):
    X_train = train_df[features]
    y_train = train_df[TARGET]

    X_test = test_df[features]
    y_test = test_df[TARGET]

    results = []
    trained_models = {}

    for name, regressor in get_models().items():
        print(f"\nTraining {name}...")

        # Create a fresh preprocessor for each model.
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(features)),
            ("model", regressor),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test, predictions
        )

        rmse = float(np.sqrt(
            mean_squared_error(y_test, predictions)
        ))

        r2 = (
            r2_score(y_test, predictions)
            if len(y_test) >= 2
            else np.nan
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        trained_models[name] = pipeline

        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(
            f"R2:   {r2:.4f}"
            if np.isfinite(r2)
            else "R2:   undefined"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "RMSE",
        ascending=True,
    ).reset_index(drop=True)

    print("\nModel comparison")
    print("----------------")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print("\nBest model based on test RMSE:", best_model_name)

    return results_df, best_model_name, best_model


# --------------------------------------------------
# 8. SAVE MODEL AND RESULTS
# --------------------------------------------------

def save_artifacts(
    results_df,
    best_model_name,
    best_model,
    features,
    train_df,
    test_df,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = MODEL_DIR / "model_comparison.csv"
    model_path = MODEL_DIR / "best_model.pkl"
    features_path = MODEL_DIR / "features.pkl"
    metadata_path = MODEL_DIR / "model_metadata.pkl"

    results_df.to_csv(
        comparison_path,
        index=False,
    )

    joblib.dump(best_model, model_path)
    joblib.dump(features, features_path)

    metadata = {
        "best_model": best_model_name,
        "features": features,
        "target": TARGET,
        "training_year_min": int(train_df["Year"].min()),
        "training_year_max": int(train_df["Year"].max()),
        "testing_year_min": int(test_df["Year"].min()),
        "testing_year_max": int(test_df["Year"].max()),
        "test_fraction": TEST_FRACTION,
        "random_state": RANDOM_STATE,
        "excluded_features": sorted(EXCLUDED_FEATURES),
    }

    joblib.dump(metadata, metadata_path)

    print("\nSaved artifacts")
    print("---------------")
    print("Best model:", model_path)
    print("Feature list:", features_path)
    print("Metadata:", metadata_path)
    print("Comparison:", comparison_path)


# --------------------------------------------------
# 9. MAIN
# --------------------------------------------------

def main():
    print("Loading notebook-preprocessed training dataset...")

    df = load_data()
    df = standardize_features(df)

    df, features = select_features(df)

    print("\nDataset shape:", df.shape)
    print("Year range:", df["Year"].min(), "-", df["Year"].max())
    print("Target:", TARGET)
    print("Features:", features)

    train_df, test_df = chronological_split(df)

    results_df, best_model_name, best_model = train_models(
        train_df,
        test_df,
        features,
    )

    save_artifacts(
        results_df,
        best_model_name,
        best_model,
        features,
        train_df,
        test_df,
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
