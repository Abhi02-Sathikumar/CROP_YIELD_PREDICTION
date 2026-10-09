import os
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. CLEAN CROP PRODUCTION DATA
# --------------------------------------------------
def clean_crop_data(path):
    """
    Clean crop production data.

    Expected columns include:
        District, State, Crop, Season, Year,
        Area, Production
    """

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # Standardize column names
    rename_map = {
        "District_Name": "District",
        "district": "District",
        "State_Name": "State",
        "state": "State",
        "Crop_Year": "Year",
        "crop_year": "Year",
        "year": "Year",
        "crop": "Crop",
        "season": "Season",
        "area": "Area",
        "production": "Production"
    }

    df = df.rename(columns=rename_map)

    required = [
        "District",
        "Crop",
        "Season",
        "Year",
        "Area",
        "Production"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing crop columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Standardize text columns
    for col in ["District", "Crop", "Season"]:
        df[col] = df[col].astype("string").str.strip()

    if "State" in df.columns:
        df["State"] = df["State"].astype("string").str.strip()

    # Convert numeric columns
    for col in ["Area", "Production", "Year"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Remove invalid records
    df = df.dropna(
        subset=[
            "District",
            "Crop",
            "Season",
            "Year",
            "Area",
            "Production"
        ]
    ).copy()

    # Keep valid years and production values
    df["Year"] = df["Year"].round().astype(int)
    df = df[df["Area"] > 0]
    df = df[df["Production"] >= 0]

    # Calculate crop yield
    df["Yield"] = df["Production"] / df["Area"]

    # Remove infinite and invalid yield values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Yield"])

    return df


# --------------------------------------------------
# 2. CLEAN SOIL DATA
# --------------------------------------------------
def clean_soil_data(path):
    """
    Clean soil/crop recommendation data.

    Supports columns such as:
        N, P, K, temperature, humidity,
        ph, rainfall, label
    """

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    rename_map = {
        "nitrogen": "N",
        "n": "N",
        "phosphorous": "P",
        "phosphorus": "P",
        "p": "P",
        "potassium": "K",
        "k": "K",
        "temperature": "soil_temperature",
        "humidity": "soil_humidity",
        "ph": "pH",
        "rainfall": "soil_rainfall",
        "label": "Crop",
        "crop": "Crop"
    }

    df = df.rename(columns=rename_map)

    if "Crop" in df.columns:
        df["Crop"] = (
            df["Crop"]
            .astype("string")
            .str.strip()
        )

    # Convert available soil measurements to numeric
    numeric_columns = [
        "N",
        "P",
        "K",
        "pH",
        "soil_temperature",
        "soil_humidity",
        "soil_rainfall"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    return df


# --------------------------------------------------
# 3. CLEAN PREPROCESSED WEATHER DATA
# --------------------------------------------------
def clean_weather_data(path):
    """
    Process state-level weather_preprocessed.csv.

    Expected source columns include:
        state, year, temp_max_c, temp_min_c,
        precipitation_mm, rain_mm, etc.

    Weather data is state-level, not district-level.
    """

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Rename geographic and year columns
    df = df.rename(
        columns={
            "state": "State",
            "year": "Year"
        }
    )

    required = ["State", "Year"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing weather columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Standardize state names
    df["State"] = (
        df["State"]
        .astype("string")
        .str.strip()
    )

    # Convert year to numeric
    df["Year"] = pd.to_numeric(
        df["Year"], errors="coerce"
    )

    df = df.dropna(
        subset=["State", "Year"]
    ).copy()

    # IMPORTANT:
    # This rounds decimal years to the nearest integer.
    # Confirm the meaning of decimal years in the source
    # before relying on this for a production model.
    df["Year"] = df["Year"].round().astype(int)

    # Convert all weather measurements to numeric
    weather_columns = [
        col for col in df.columns
        if col not in ["State", "Year"]
    ]

    for col in weather_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Aggregate duplicate state-year observations
    # to ensure one weather row per merge key
    df = (
        df.groupby(
            ["State", "Year"],
            as_index=False
        )[weather_columns]
        .mean()
    )

    return df


# --------------------------------------------------
# 4. MERGE CROP, WEATHER AND SOIL DATA
# --------------------------------------------------
def merge_datasets(crop_df, weather_df, soil_df):
    """
    Merge:
        Crop + weather using State and Year
        Crop + soil using Crop

    State-level weather will be shared by districts
    belonging to the same state and year.
    """

    crop_df = crop_df.copy()
    weather_df = weather_df.copy()
    soil_df = soil_df.copy()

    # Normalize state names for matching
    if (
        "State" in crop_df.columns
        and "State" in weather_df.columns
    ):
        crop_df["State"] = (
            crop_df["State"]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        weather_df["State"] = (
            weather_df["State"]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        # Merge using state and year
        merged = crop_df.merge(
            weather_df,
            on=["State", "Year"],
            how="left",
            validate="many_to_one"
        )

    elif "District" in weather_df.columns:
        # Fallback for a district-level weather dataset
        merged = crop_df.merge(
            weather_df,
            on=["District", "Year"],
            how="left",
            validate="many_to_one"
        )

    else:
        raise ValueError(
            "Cannot merge crop and weather data. "
            "The crop dataset needs State and Year "
            "to match the state-level weather dataset."
        )

    # Merge soil data by crop, if available
    if "Crop" in soil_df.columns:

        soil_columns = [
            col for col in [
                "Crop",
                "N",
                "P",
                "K",
                "pH"
            ]
            if col in soil_df.columns
        ]

        if len(soil_columns) > 1:
            soil_small = soil_df[soil_columns].copy()

            # Normalize crop labels for matching
            soil_small["Crop"] = (
                soil_small["Crop"]
                .astype("string")
                .str.strip()
                .str.casefold()
            )

            merged["Crop"] = (
                merged["Crop"]
                .astype("string")
                .str.strip()
                .str.casefold()
            )

            # Average soil properties by crop
            soil_small = (
                soil_small
                .groupby("Crop", as_index=False)
                .mean(numeric_only=True)
            )

            merged = merged.merge(
                soil_small,
                on="Crop",
                how="left",
                validate="many_to_one"
            )

    return merged


# --------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":

    # Project root is the current working directory.
    # Run this script from the repository root.

    crop_path = "data/raw/crop_production.csv"

    soil_path = "data/raw/crop_recommendation.csv"

    # UPDATED: Read the preprocessed weather CSV
    weather_path = (
        "data/processed_data/weather_preprocessed.csv"
    )

    output_path = "data/training_data.csv"

    # Verify input files exist
    for path in [
        crop_path,
        soil_path,
        weather_path
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Input file not found: {path}\n"
                "Run this script from the project root "
                "and verify the file path."
            )

    print("Loading and cleaning crop data...")
    crop = clean_crop_data(crop_path)

    print("Loading and cleaning soil data...")
    soil = clean_soil_data(soil_path)

    print("Loading preprocessed weather data...")
    weather = clean_weather_data(weather_path)

    print("\nCrop data shape:", crop.shape)
    print("Soil data shape:", soil.shape)
    print("Weather data shape:", weather.shape)

    print("\nWeather columns:")
    print(weather.columns.tolist())

    print("\nMerging datasets...")
    final = merge_datasets(
        crop,
        weather,
        soil
    )

    # Create output directory if needed
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # Save final training dataset
    final.to_csv(
        output_path,
        index=False
    )

    print("\nPreprocessing completed successfully.")
    print("Output saved to:", output_path)

    print("\nFinal dataset preview:")
    print(final.head())

    print("\nFinal dataset shape:")
    print(final.shape)

    print("\nFinal columns:")
    print(final.columns.tolist())

    # Report weather missing-value percentages
    weather_columns = [
        col for col in weather.columns
        if col not in ["State", "Year"]
    ]

    print("\nWeather missing-value percentages:")
    if weather_columns:
        print(
            (
                final[weather_columns]
                .isna()
                .mean()
                .mul(100)
                .round(2)
                .astype(str)
                + "%"
            )
        )

    # Report the number of rows without matched weather
    if weather_columns:
        unmatched = final[weather_columns].isna().all(axis=1).sum()

        print(
            "\nRows with no matched weather data:",
            unmatched,
            "out of",
            len(final)
        )
