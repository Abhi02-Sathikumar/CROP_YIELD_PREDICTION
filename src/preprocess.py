from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_data"
OUTPUT_DIR = PROJECT_ROOT / "data"

CROP_PATH = PROCESSED_DIR / "crop_production_preprocessed.csv"
WEATHER_PATH = PROCESSED_DIR / "weather_state_year_preprocessed.csv"
RECOMMENDATION_PATH = (
    PROCESSED_DIR / "crop_recommendation_preprocessed.csv"
)

YIELD_OUTPUT_PATH = OUTPUT_DIR / "training_data.csv"
RECOMMENDATION_OUTPUT_PATH = (
    OUTPUT_DIR / "crop_recommendation_training_data.csv"
)


# --------------------------------------------------
# 2. SHARED HELPERS
# --------------------------------------------------

def require_file(path):
    """Raise a clear error when a notebook output is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required notebook output not found:\n{path}\n"
            "Run the relevant preprocessing notebook first."
        )


def normalize_state(series):
    """Normalize state names for reliable joins."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
    )


def normalize_crop(series):
    """Normalize crop labels without changing their meaning."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
    )


# --------------------------------------------------
# 3. LOAD NOTEBOOK-PREPROCESSED CROP PRODUCTION DATA
# --------------------------------------------------

def load_crop_data(path=CROP_PATH):
    require_file(path)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    rename_map = {
        "State_Name": "State",
        "state_name": "State",
        "state": "State",
        "District_Name": "District",
        "district_name": "District",
        "district": "District",
        "Crop_Year": "Year",
        "crop_year": "Year",
        "year": "Year",
        "crop": "Crop",
        "season": "Season",
        "area": "Area",
        "production": "Production",
        "yield": "Yield",
    }

    df = df.rename(columns=rename_map)

    required = [
        "State",
        "District",
        "Crop",
        "Season",
        "Year",
        "Area",
        "Production",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Crop production file is missing {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    for col in ["State", "District", "Crop", "Season"]:
        df[col] = df[col].astype("string").str.strip()

    for col in ["Year", "Area", "Production"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()

    # Crop_Year should represent an actual calendar year.
    df = df[df["Year"] % 1 == 0].copy()
    df["Year"] = df["Year"].astype(int)

    df = df[
        (df["Area"] > 0) &
        (df["Production"] >= 0)
    ].copy()

    # Recalculate the target consistently.
    # This is production per unit of cultivated area.
    df["Yield"] = df["Production"] / df["Area"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Yield"]).copy()

    df["State"] = normalize_state(df["State"])

    print("Crop production data:", df.shape)
    return df


# --------------------------------------------------
# 4. LOAD NOTEBOOK-PREPROCESSED WEATHER DATA
# --------------------------------------------------

def load_weather_data(path=WEATHER_PATH):
    require_file(path)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    rename_map = {
        "state": "State",
        "year": "Year",
    }
    df = df.rename(columns=rename_map)

    required = ["State", "Year"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Weather file is missing {missing}. "
            "Export weather aggregated by both state and year."
        )

    df["State"] = df["State"].astype("string").str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    df = df.dropna(subset=["State", "Year"]).copy()

    # Do not round averaged years. Reject non-integer year keys.
    df = df[df["Year"] % 1 == 0].copy()
    df["Year"] = df["Year"].astype(int)

    df["State"] = normalize_state(df["State"])

    weather_columns = [
        col for col in df.columns
        if col not in ["State", "Year"]
    ]

    if not weather_columns:
        raise ValueError("Weather file has no usable weather features.")

    for col in weather_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # One weather row per state-year is essential for a safe merge.
    if df.duplicated(["State", "Year"]).any():
        duplicates = df.loc[
            df.duplicated(["State", "Year"], keep=False),
            ["State", "Year"]
        ].drop_duplicates()

        raise ValueError(
            "Weather data has duplicate state-year keys. "
            "Fix the aggregation in Weather_data.ipynb first.\n"
            f"Examples:\n{duplicates.head()}"
        )

    print("Weather data:", df.shape)
    return df


# --------------------------------------------------
# 5. PREPARE SEPARATE CROP RECOMMENDATION DATA
# --------------------------------------------------

def load_recommendation_data(path=RECOMMENDATION_PATH):
    require_file(path)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {
        "ph": "pH",
        "label": "Crop",
    }
    df = df.rename(columns=rename_map)

    features = [
        "N", "P", "K", "temperature",
        "humidity", "pH", "rainfall"
    ]

    missing = [col for col in features + ["Crop"] if col not in df.columns]
    if missing:
        raise ValueError(
            f"Recommendation file is missing {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Crop"] = df["Crop"].astype("string").str.strip()
    df = df.dropna(subset=features + ["Crop"]).copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + ["Crop"])

    print("Crop recommendation data:", df.shape)
    return df


# --------------------------------------------------
# 6. MERGE PRODUCTION AND WEATHER
# --------------------------------------------------

def merge_yield_data(crop_df, weather_df):
    merged = crop_df.merge(
        weather_df,
        on=["State", "Year"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = int((merged["_merge"] == "left_only").sum())
    total = len(merged)

    print(f"Production records: {total}")
    print(f"Records with no matching weather: {unmatched}")

    merged = merged.drop(columns=["_merge"])

    # Do not silently fill missing weather with zero.
    weather_columns = [
        col for col in weather_df.columns
        if col not in ["State", "Year"]
    ]

    if weather_columns:
        print("\nWeather missing-value percentages:")
        print(
            (merged[weather_columns].isna().mean() * 100)
            .round(2)
            .astype(str)
            .add("%")
        )

    return merged


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading notebook-preprocessed datasets...")

    crop_df = load_crop_data()
    weather_df = load_weather_data()
    recommendation_df = load_recommendation_data()

    print("\nMerging crop production with yearly weather...")
    yield_df = merge_yield_data(crop_df, weather_df)

    # Keep both model datasets separate.
    yield_df.to_csv(YIELD_OUTPUT_PATH, index=False)
    recommendation_df.to_csv(
        RECOMMENDATION_OUTPUT_PATH,
        index=False
    )

    print("\nPreprocessing completed.")
    print("Yield training data:", YIELD_OUTPUT_PATH)
    print("Yield dataset shape:", yield_df.shape)
    print("Yield columns:", yield_df.columns.tolist())

    print(
        "\nCrop recommendation training data:",
        RECOMMENDATION_OUTPUT_PATH
    )
    print("Recommendation dataset shape:", recommendation_df.shape)


if __name__ == "__main__":
    main()
