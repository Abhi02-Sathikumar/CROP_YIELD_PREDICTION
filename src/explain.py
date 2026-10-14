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

DATA_PATH = ROOT / "data" / "processed_data" / "crop_production_preprocessed.csv"
MODEL_PATH = ROOT / "models" / "xgboost.joblib"
OUTPUT_DIR = ROOT / "outputs" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset and trained XGBoost model...")

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={
    "State_Name": "State",
    "District_Name": "District",
    "Crop_Year": "Year"
})

features = ["State", "District", "Year", "Season", "Crop", "Area"]

for col in ["Year", "Area", "Production", "Yield"]:
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

# Use a sample to make SHAP analysis faster.
sample_size = min(300, len(df))
sample = df.sample(n=sample_size, random_state=42)

X_sample = sample[features]

pipeline = joblib.load(MODEL_PATH)
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

X_transformed = preprocessor.transform(X_sample)

# SHAP works with a dense array for this explanation workflow.
if sparse.issparse(X_transformed):
    X_transformed = X_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

print("Calculating SHAP values for", sample_size, "records...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# Save a SHAP summary plot.
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# Save a global feature-importance bar chart.
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
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# Save a CSV ranking features by mean absolute SHAP value.
importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": np.abs(shap_values).mean(axis=0)
}).sort_values("Mean_Absolute_SHAP", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTop 15 influential features:")
print(importance.head(15).to_string(index=False))

print("\nSHAP analysis completed.")
print("Results saved in:", OUTPUT_DIR)
