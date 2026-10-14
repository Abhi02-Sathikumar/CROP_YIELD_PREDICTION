from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    raise SystemExit("Install XGBoost using: python -m pip install xgboost")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed_data" / "crop_weather_merged.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs" / "weather_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading merged crop-weather data...")
df = pd.read_csv(DATA_PATH)

# Convert required columns to numeric values.
numeric_cols = [
    "Year", "Area", "Production", "Yield",
    "Mean_Temperature", "Avg_Max_Temp",
    "Avg_Min_Temp", "Total_Precipitation_mm",
    "Max_Wind_Speed"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

required_cols = [
    "State", "District", "Season", "Crop",
    "Year", "Area", "Production", "Yield",
    "Mean_Temperature", "Avg_Max_Temp",
    "Avg_Min_Temp", "Total_Precipitation_mm",
    "Max_Wind_Speed"
]

df = df.dropna(subset=required_cols).copy()
df = df[
    (df["Area"] > 0)
    & (df["Production"] >= 0)
    & (df["Yield"] >= 0)
]

# Do not include Production as a feature because yield is
# derived from production and area.
categorical_features = ["State", "District", "Season", "Crop"]
numeric_features = [
    "Year", "Area", "Mean_Temperature",
    "Avg_Max_Temp", "Avg_Min_Temp",
    "Total_Precipitation_mm", "Max_Wind_Speed"
]
features = categorical_features + numeric_features
target = "Yield"

X = df[features]
y = df[target]

# Keep all records from a year together to reduce temporal leakage.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=df["Year"])
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training years:", sorted(X_train["Year"].unique()))
print("Testing years:", sorted(X_test["Year"].unique()))

preprocessor = ColumnTransformer([
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    ),
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numeric_features
    )
])

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=120,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name} with weather features...")
    start = time.time()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    elapsed = time.time() - start

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Training_Time_Seconds": elapsed
    })

    filename = (
        "random_forest_weather.joblib"
        if name == "Random Forest"
        else "xgboost_weather.joblib"
    )

    joblib.dump(pipeline, MODEL_DIR / filename)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Training time: {elapsed:.1f} seconds")
    print("Saved:", MODEL_DIR / filename)

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(OUTPUT_DIR / "model_comparison_weather.csv", index=False)

print("\nWEATHER MODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nSaved comparison:", OUTPUT_DIR / "model_comparison_weather.csv")
print("\nWeather model training completed.")
