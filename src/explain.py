"""Explain crop yield predictions with SHAP."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def load_artifacts(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    model_path = model_dir / "best_model.pkl"
    features_path = model_dir / "features.pkl"

    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError(
            "Trained artifacts not found. Run src/train.py first."
        )

    return joblib.load(model_path), joblib.load(features_path)


def explain_yield(
    input_data,
    background_data,
    model=None,
    features=None,
    max_background_rows=100,
):
    """
    Return SHAP contributions for one yield prediction.

    input_data: one-row DataFrame, Series, or dictionary.
    background_data: processed training DataFrame with the original
                     feature columns used by the model.

    Positive SHAP values push the prediction higher; negative values
    push it lower relative to the SHAP baseline.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "Install SHAP first with: pip install shap"
        ) from exc

    if model is None or features is None:
        model, features = load_artifacts()

    if isinstance(input_data, dict):
        row = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        row = input_data.to_frame().T
    elif isinstance(input_data, pd.DataFrame) and len(input_data) == 1:
        row = input_data.copy()
    else:
        raise ValueError("input_data must contain exactly one row.")

    row.columns = row.columns.astype(str).str.strip()
    background_data = background_data.copy()
    background_data.columns = (
        background_data.columns.astype(str).str.strip()
    )

    # The training pipeline expects the same raw input columns and order.
    row = row.reindex(columns=features)
    background = background_data.reindex(columns=features)

    background = background.replace([np.inf, -np.inf], np.nan)
    row = row.replace([np.inf, -np.inf], np.nan)

    background = background.dropna(how="all")
    if background.empty:
        raise ValueError(
            "background_data has no usable rows for SHAP."
        )

    if len(background) > max_background_rows:
        background = background.sample(
            n=max_background_rows, random_state=42
        )

    # Apply the fitted transformations; do not refit the encoder/imputer.
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    transformed_background = preprocessor.transform(background)
    transformed_row = preprocessor.transform(row)

    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()
    if hasattr(transformed_row, "toarray"):
        transformed_row = transformed_row.toarray()

    try:
        transformed_names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        transformed_names = [
            f"feature_{i}" for i in range(transformed_row.shape[1])
        ]

    # A model-agnostic explainer supports more estimator types.
    # A small background sample limits computation time.
    explainer = shap.Explainer(
        estimator,
        transformed_background,
        feature_names=list(transformed_names),
    )
    explanation = explainer(transformed_row)

    shap_values = np.asarray(explanation.values)
    if shap_values.ndim == 2:
        contributions = shap_values[0]
    elif shap_values.ndim == 3:
        contributions = shap_values[0, :, 0]
    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    base_values = np.asarray(explanation.base_values).reshape(-1)
    baseline = float(base_values[0])
    prediction = float(np.asarray(model.predict(row)).reshape(-1)[0])

    result = pd.DataFrame({
        "feature": list(transformed_names),
        "feature_value": transformed_row[0],
        "shap_value": contributions,
    })

    result["direction"] = np.where(
        result["shap_value"] > 0,
        "increases prediction",
        np.where(
            result["shap_value"] < 0,
            "decreases prediction",
            "no contribution",
        ),
    )

    result["absolute_impact"] = result["shap_value"].abs()
    result = result.sort_values(
        "absolute_impact", ascending=False
    ).reset_index(drop=True)

    return {
        "predicted_yield": round(prediction, 4),
        "baseline_yield": round(baseline, 4),
        "explanations": result,
        "note": (
            "SHAP values are relative to the background sample. "
            "They explain model behaviour, not causal effects."
        ),
    }


if __name__ == "__main__":
    print(
        "Import explain_yield() from a notebook and pass a "
        "one-row input DataFrame plus a representative training "
        "DataFrame as background_data."
    )