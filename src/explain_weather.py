from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed_data" / "crop_weather_merged.csv"
MODEL_PATH = ROOT / "models" / "xgboost_weather.joblib"
OUTPUT_DIR = ROOT / "outputs" / "shap_weather"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading weather-trained XGBoost model...")
pipeline = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

categorical_features = ["State", "District", "Season", "Crop"]
numeric_features = [
    "Year", "Area", "Mean_Temperature",
    "Avg_Max_Temp", "Avg_Min_Temp",
    "Total_Precipitation_mm", "Max_Wind_Speed"
]
features = categorical_features + numeric_features

for col in numeric_features + ["Production", "Yield"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(
    subset=features + ["Production", "Yield"]
).copy()
df = df[
    (df["Area"] > 0)
    & (df["Production"] >= 0)
    & (df["Yield"] >= 0)
]

# Use a manageable random sample for SHAP.
sample = df.sample(n=min(300, len(df)), random_state=42)
X_sample = sample[features]

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

print("Transforming sample...")
X_transformed = preprocessor.transform(X_sample)

if sparse.issparse(X_transformed):
    X_transformed = X_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# Summary plot: direction and magnitude of feature influence.
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary_weather.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# Global feature importance.
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_importance_weather.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": np.abs(shap_values).mean(axis=0)
}).sort_values("Mean_Absolute_SHAP", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "shap_importance_weather.csv",
    index=False
)

print("\nTop 20 features:")
print(importance.head(20).to_string(index=False))

print("\nSHAP analysis completed.")
print("Results saved to:", OUTPUT_DIR)
