import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


DATA_PATH = "data/training_data.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()


# --------------------------------------------------
# Basic cleaning
# --------------------------------------------------

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=["Yield"]
)


# --------------------------------------------------
# Features
# --------------------------------------------------

features = [
    "Crop",
    "District",
    "Season",
    "Rainfall",
    "AvgTemperature",
    "AvgHumidity",
    "N",
    "P",
    "K",
    "pH"
]

features = [
    x for x in features
    if x in df.columns
]

target = "Yield"


print("Features:")
print(features)


# --------------------------------------------------
# Remove rows with excessive missing values
# --------------------------------------------------

df = df.dropna(
    subset=features,
    how="all"
)


# --------------------------------------------------
# Time based split
# --------------------------------------------------

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df = df.dropna(
    subset=["Year"]
)

df["Year"] = df["Year"].astype(int)


train_df = df[
    df["Year"] <= 2019
]

test_df = df[
    df["Year"] > 2019
]

print(
    "Training rows:",
    len(train_df)
)

print(
    "Testing rows:",
    len(test_df)
)


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

categorical_features = [
    x for x in [
        "Crop",
        "District",
        "Season"
    ]
    if x in features
]

numeric_features = [
    x for x in features
    if x not in categorical_features
]


preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),

            categorical_features
        ),

        (
            "numeric",

            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]),

            numeric_features
        )
    ]
)


# --------------------------------------------------
# Models
# --------------------------------------------------

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )
}


results = {}

trained_models = {}


# --------------------------------------------------
# Train
# --------------------------------------------------

for name, regressor in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            regressor
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    trained_models[name] = pipeline

    print(
        f"MAE: {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R²: {r2:.4f}"
    )


# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(
    results
).T

print("\nModel comparison:")
print(results_df)

results_df.to_csv(
    "models/model_comparison.csv"
)


# --------------------------------------------------
# Find best model
# --------------------------------------------------

best_model_name = (
    results_df["RMSE"]
    .idxmin()
)

best_model = trained_models[
    best_model_name
]

print(
    "\nBest model:",
    best_model_name
)


# --------------------------------------------------
# Save
# --------------------------------------------------

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

joblib.dump(
    features,
    "models/features.pkl"
)

print(
    "\nModel saved successfully."
)

