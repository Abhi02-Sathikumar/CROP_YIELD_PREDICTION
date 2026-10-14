from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CROP_PATH = ROOT / "data" / "processed_data" / "crop_production_preprocessed.csv"
WEATHER_PATH = ROOT / "data" / "raw" / "india_states_yearly_weather_2000_2014.csv"
OUTPUT_DIR = ROOT / "data" / "processed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading crop and weather datasets...")
crop = pd.read_csv(CROP_PATH)
weather = pd.read_csv(WEATHER_PATH)

crop = crop.rename(columns={
    "State_Name": "State",
    "District_Name": "District",
    "Crop_Year": "Year"
})

# Convert state names to a consistent format.
def normalize_state(value):
    return " ".join(str(value).strip().upper().split())

aliases = {
    "ANDAMAN AND NICOBAR": "ANDAMAN AND NICOBAR ISLANDS",
    "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
    "ORISSA": "ODISHA",
    "ODISHA": "ODISHA",
    "UTTARANCHAL": "UTTARAKHAND",
    "UTTARAKHAND": "UTTARAKHAND",
}

crop["State_Key"] = crop["State"].map(normalize_state).replace(aliases)
weather["State_Key"] = weather["State"].map(normalize_state).replace(aliases)

crop["Year"] = pd.to_numeric(crop["Year"], errors="coerce")
weather["Year"] = pd.to_numeric(weather["Year"], errors="coerce")

# Keep only the years available in both datasets.
crop = crop[crop["Year"].between(2000, 2014)].copy()
weather = weather[weather["Year"].between(2000, 2014)].copy()

weather_columns = [
    "Mean_Temperature",
    "Avg_Max_Temp",
    "Avg_Min_Temp",
    "Total_Precipitation_mm",
    "Max_Wind_Speed",
]

# Check that each weather state-year appears only once.
duplicates = weather.duplicated(["State_Key", "Year"], keep=False)
if duplicates.any():
    print("Duplicate weather state-year records found:")
    print(weather.loc[duplicates, ["State", "Year"]].to_string(index=False))
    raise ValueError("Resolve duplicate weather state-year records before merging.")

weather_small = weather[
    ["State_Key", "Year"] + weather_columns
].copy()

print("Crop records in overlapping years:", len(crop))
print("Weather records:", len(weather_small))

# Keep all crop records initially so unmatched rows can be counted.
merged = crop.merge(
    weather_small,
    on=["State_Key", "Year"],
    how="left",
    validate="many_to_one",
    indicator=True
)

matched = merged[merged["_merge"] == "both"].copy()
unmatched = merged[merged["_merge"] == "left_only"].copy()

# Save the complete merge for inspection.
merged.drop(columns=["State_Key", "_merge"]).to_csv(
    OUTPUT_DIR / "crop_weather_merge_all.csv", index=False
)

# Save matched records for weather analysis and modeling.
matched.drop(columns=["State_Key", "_merge"]).to_csv(
    OUTPUT_DIR / "crop_weather_merged.csv", index=False
)

print("\nMERGE RESULTS")
print("Total crop records:", len(merged))
print("Matched records:", len(matched))
print("Unmatched records:", len(unmatched))

if len(merged):
    print(f"Match rate: {len(matched) / len(merged) * 100:.2f}%")

if not unmatched.empty:
    print("\nUnmatched crop state-year examples:")
    print(
        unmatched[["State", "Year"]]
        .drop_duplicates()
        .head(20)
        .to_string(index=False)
    )

print("\nWeather columns missing in matched dataset:")
print(matched[weather_columns].isna().sum().to_string())

print("\nSaved files:")
print(OUTPUT_DIR / "crop_weather_merge_all.csv")
print(OUTPUT_DIR / "crop_weather_merged.csv")
