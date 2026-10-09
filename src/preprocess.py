
import pandas as pd
import numpy as np


def clean_crop_data(path):
    """
    Load and clean crop production data.

    Expected columns:
        District
        State
        Crop
        Season
        Year
        Area
        Production
    """

    df = pd.read_csv(path)

    # Remove whitespace from column names
    df.columns = df.columns.str.strip()

    # Standardize column names
    rename_map = {
        "District_Name": "District",
        "district": "District",
        "State_Name": "State",
        "state": "State",
        "Crop_Year": "Year",
        "crop": "Crop",
        "Area": "Area",
        "Production": "Production"
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

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Convert numeric fields
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df["Production"] = pd.to_numeric(
        df["Production"],
        errors="coerce"
    )

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce"
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
    )

    df = df[df["Area"] > 0]
    df = df[df["Production"] >= 0]

    # Yield
    df["Yield"] = (
        df["Production"] / df["Area"]
    )

    # Remove extreme invalid values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(subset=["Yield"])

    return df


def clean_soil_data(path):
    """
    Expected columns:
        N
        P
        K
        temperature
        humidity
        ph
        rainfall

    This dataset is often a crop recommendation
    dataset rather than a district-level soil dataset.
    """

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    rename_map = {
        "nitrogen": "N",
        "phosphorous": "P",
        "phosphorus": "P",
        "potassium": "K",
        "temperature": "soil_temperature",
        "humidity": "soil_humidity",
        "ph": "pH",
        "rainfall": "soil_rainfall",
        "label": "Crop"
    }

    df = df.rename(columns=rename_map)

    # Standardize crop name
    if "Crop" in df.columns:
        df["Crop"] = (
            df["Crop"]
            .astype(str)
            .str.strip()
        )

    return df


def clean_weather_data(path):
    """
    Expected columns:

        District
        Year
        Rainfall
        AvgTemperature
        Humidity
    """

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    return df


def merge_datasets(
    crop_df,
    weather_df,
    soil_df
):
    """
    Merge crop, weather and soil datasets.

    Weather should ideally have:
        District
        Year

    Soil should ideally have:
        Crop
    """

    # Crop + weather
    merged = crop_df.merge(
        weather_df,
        on=["District", "Year"],
        how="left"
    )

    # Crop + soil
    if "Crop" in soil_df.columns:

        soil_columns = [
            c for c in [
                "Crop",
                "N",
                "P",
                "K",
                "pH"
            ]
            if c in soil_df.columns
        ]

        soil_small = soil_df[soil_columns].copy()

        # Average soil properties by crop
        soil_small = (
            soil_small
            .groupby("Crop")
            .mean(numeric_only=True)
            .reset_index()
        )

        merged = merged.merge(
            soil_small,
            on="Crop",
            how="left"
        )

    return merged


if __name__ == "__main__":

    crop = clean_crop_data(
        "data/raw/crop_production.csv"
    )

    soil = clean_soil_data(
        "data/raw/crop_recommendation.csv"
    )

    weather = clean_weather_data(
        "data/raw/weather.csv"
    )

    final = merge_datasets(
        crop,
        weather,
        soil
    )

    final.to_csv(
        "data/training_data.csv",
        index=False
    )

    print("Final dataset:")
    print(final.head())

    print("\nShape:")
    print(final.shape)

    print("\nColumns:")
    print(final.columns.tolist())
    

