from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    raise SystemExit(
        "XGBoost is missing. Run: python -m pip install xgboost"
    )

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed_data" / "crop_production_preprocessed.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading crop production dataset...")
df = pd.read_csv(DATA_PATH)

df = df.rename(columns={
    "State_Name": "State",
    "District_Name": "District",
    "Crop_Year": "Year"
})

required = [
    "State", "District", "Year", "Season",
    "Crop", "Area", "Production", "Yield"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()

for col in ["Year", "Area", "Production", "Yield"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Year", "Area", "Production", "Yield"])
df = df[(df["Area"] > 0) & (df["Production"] >= 0) & (df["Yield"] >= 0)]

# Production is deliberately excluded because Yield is derived from
# Production / Area; including Production would cause target leakage.
features = ["State", "District", "Year", "Season", "Crop", "Area"]
target = "Yield"

X = df[features]
y = df[target]

categorical = ["State", "District", "Season", "Crop"]
numeric = ["Year", "Area"]

preprocessor = ColumnTransformer([
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
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
    print(f"\nTraining {name}...")
    start = time.time()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    seconds = time.time() - start

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Training_Time_Seconds": seconds
    })

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, MODEL_DIR / filename)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Training time: {seconds:.1f} seconds")
    print("Saved:", MODEL_DIR / filename)

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nSaved comparison to:", MODEL_DIR / "model_comparison.csv")
print("\nTraining completed.")
