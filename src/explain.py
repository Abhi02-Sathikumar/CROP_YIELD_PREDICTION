from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed_data"
    / "crop_production_preprocessed.csv"
)

MODEL_PATHS = {
    "random_forest": ROOT / "models" / "random_forest.joblib",
    "xgboost": ROOT / "models" / "xgboost.joblib",
}

OUTPUT_DIR = ROOT / "outputs" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "State",
    "District",
    "Year",
    "Season",
    "Crop",
    "Area",
]

SAMPLE_SIZE = 300
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load and clean the dataset
# --------------------------------------------------

print("Loading crop dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

df = df.rename(
    columns={
        "State_Name": "State",
        "District_Name": "District",
        "Crop_Year": "Year",
    }
)

required_columns = FEATURES + ["Production", "Yield"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required dataset columns: {missing_columns}"
    )

for col in ["Year", "Area", "Production", "Yield"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=required_columns).copy()

df = df[
    (df["Area"] > 0)
    & (df["Production"] >= 0)
    & (df["Yield"] >= 0)
].copy()

if df.empty:
    raise ValueError("No valid records remain after data cleaning.")

sample_size = min(SAMPLE_SIZE, len(df))

sample = df.sample(
    n=sample_size,
    random_state=RANDOM_STATE,
)

X_sample = sample[FEATURES]

print(f"Records available: {len(df)}")
print(f"Records selected for SHAP: {sample_size}")


# --------------------------------------------------
# 3. Explain both trained models
# --------------------------------------------------

successful_models = []
missing_models = []

for model_name, model_path in MODEL_PATHS.items():

    print(f"\n{'=' * 55}")
    print(f"Explaining {model_name.upper()}")
    print(f"{'=' * 55}")

    if not model_path.exists():
        print(f"WARNING: Model not found: {model_path}")
        missing_models.append(model_name)
        continue

    try:
        # Load the saved preprocessing + model pipeline.
        pipeline = joblib.load(model_path)

        if not hasattr(pipeline, "named_steps"):
            raise ValueError(
                "Expected a saved sklearn-style pipeline."
            )

        if (
            "preprocessor" not in pipeline.named_steps
            or "model" not in pipeline.named_steps
        ):
            raise ValueError(
                "Pipeline must contain 'preprocessor' "
                "and 'model' steps."
            )

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        # Apply exactly the saved preprocessing.
        X_transformed = preprocessor.transform(X_sample)

        if sparse.issparse(X_transformed):
            X_transformed = X_transformed.toarray()

        X_transformed = np.asarray(X_transformed)

        if X_transformed.ndim != 2:
            raise ValueError(
                f"Unexpected transformed data shape: "
                f"{X_transformed.shape}"
            )

        # Obtain the transformed feature names.
        try:
            feature_names = preprocessor.get_feature_names_out()
        except (AttributeError, ValueError):
            feature_names = np.array(
                [
                    f"feature_{i}"
                    for i in range(X_transformed.shape[1])
                ]
            )

        if len(feature_names) != X_transformed.shape[1]:
            feature_names = np.array(
                [
                    f"feature_{i}"
                    for i in range(X_transformed.shape[1])
                ]
            )

        print(
            f"Calculating SHAP values for {sample_size} records..."
        )

        # TreeExplainer supports tree-based estimators.
        explainer = shap.TreeExplainer(model)

        shap_result = explainer.shap_values(X_transformed)

        # Handle SHAP Explanation objects.
        if hasattr(shap_result, "values"):
            shap_values = np.asarray(shap_result.values)
        else:
            shap_values = np.asarray(shap_result)

        # Normalize output for single-output regression.
        if (
            shap_values.ndim == 3
            and shap_values.shape[-1] == 1
        ):
            shap_values = shap_values[:, :, 0]

        if shap_values.ndim != 2:
            raise ValueError(
                f"Unexpected SHAP shape: {shap_values.shape}"
            )

        if shap_values.shape != X_transformed.shape:
            raise ValueError(
                f"SHAP shape {shap_values.shape} does not match "
                f"input shape {X_transformed.shape}"
            )

        # Create a separate directory for each model.
        model_output_dir = OUTPUT_DIR / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # 4. SHAP summary plot
        # --------------------------------------------------

        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            max_display=20,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            model_output_dir / "shap_summary.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        # --------------------------------------------------
        # 5. SHAP feature importance plot
        # --------------------------------------------------

        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            plot_type="bar",
            max_display=20,
            show=False,
        )

        plt.tight_layout()

        plt.savefig(
            model_output_dir / "shap_feature_importance.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        # --------------------------------------------------
        # 6. Save feature importance CSV
        # --------------------------------------------------

        importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean_Absolute_SHAP": np.abs(
                    shap_values
                ).mean(axis=0),
            }
        ).sort_values(
            "Mean_Absolute_SHAP",
            ascending=False,
        )

        importance.to_csv(
            model_output_dir / "shap_feature_importance.csv",
            index=False,
        )

        print(f"\nTop 10 features for {model_name}:")

        print(
            importance.head(10).to_string(index=False)
        )

        print(f"Results saved to: {model_output_dir}")

        successful_models.append(model_name)

    except Exception as exc:
        print(f"ERROR explaining {model_name}: {exc}")


# --------------------------------------------------
# 7. Final status
# --------------------------------------------------

print("\n" + "=" * 55)
print("SHAP ANALYSIS SUMMARY")
print("=" * 55)

print("Successfully explained:", successful_models)

if missing_models:
    print("Missing model files:", missing_models)

if len(successful_models) == len(MODEL_PATHS):
    print("Both Random Forest and XGBoost were explained.")
elif successful_models:
    print(
        "SHAP analysis completed for available models only. "
        "The missing or failed models still need attention."
    )
else:
    print(
        "No model was successfully explained. "
        "Check the model paths and error messages above."
    )
