from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed_data" / "crop_weather_merged.csv"
OUTPUT_DIR = ROOT / "outputs" / "weather_correlation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading merged crop and weather data...")
df = pd.read_csv(DATA_PATH)

weather_features = [
    "Mean_Temperature",
    "Avg_Max_Temp",
    "Avg_Min_Temp",
    "Total_Precipitation_mm",
    "Max_Wind_Speed",
]

required = ["State", "Year", "Area", "Production", "Yield"] + weather_features
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in ["Year", "Area", "Production", "Yield"] + weather_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required)
df = df[(df["Area"] > 0) & (df["Production"] >= 0)]

# Aggregate crop records into one record per state and year.
# Weighted yield = total production / total area.
state_year = (
    df.groupby(["State", "Year"], as_index=False)
    .agg(
        Total_Area=("Area", "sum"),
        Total_Production=("Production", "sum"),
        Mean_Yield=("Yield", "mean"),
        **{col: (col, "first") for col in weather_features}
    )
)

state_year["Weighted_Yield"] = (
    state_year["Total_Production"] / state_year["Total_Area"]
)

state_year = state_year.replace([np.inf, -np.inf], np.nan)
state_year = state_year.dropna(
    subset=weather_features + ["Weighted_Yield"]
)

state_year.to_csv(
    OUTPUT_DIR / "state_year_weather_yield.csv",
    index=False
)

print("State-year observations:", len(state_year))
print("Unique states:", state_year["State"].nunique())
print("Years:", int(state_year["Year"].min()), "to", int(state_year["Year"].max()))

# Pearson correlation measures linear association.
correlations = state_year[weather_features + ["Weighted_Yield"]].corr(
    method="pearson"
)

yield_correlations = (
    correlations["Weighted_Yield"]
    .drop("Weighted_Yield")
    .sort_values(key=lambda values: values.abs(), ascending=False)
)

print("\nCorrelation of weather features with weighted yield:")
print(yield_correlations.to_string())

yield_correlations.rename("Pearson_Correlation").to_csv(
    OUTPUT_DIR / "weather_yield_correlations.csv"
)

# Draw a correlation heatmap using Matplotlib.
fig, ax = plt.subplots(figsize=(10, 7))
image = ax.imshow(correlations, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(correlations.columns)))
ax.set_xticklabels(correlations.columns, rotation=45, ha="right")
ax.set_yticks(range(len(correlations.index)))
ax.set_yticklabels(correlations.index)

for i in range(len(correlations.index)):
    for j in range(len(correlations.columns)):
        ax.text(
            j, i, f"{correlations.iloc[i, j]:.2f}",
            ha="center", va="center", fontsize=8
        )

fig.colorbar(image, ax=ax, label="Pearson correlation")
ax.set_title("State-Year Weather and Yield Correlation")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
plt.close(fig)

# Scatterplots: weather feature against weighted yield.
for feature in weather_features:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        state_year[feature],
        state_year["Weighted_Yield"],
        alpha=0.6
    )
    ax.set_xlabel(feature)
    ax.set_ylabel("Weighted yield (production / area)")
    ax.set_title(f"{feature} vs Weighted Yield")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    safe_name = feature.lower()
    fig.savefig(
        OUTPUT_DIR / f"{safe_name}_vs_yield.png",
        dpi=200
    )
    plt.close(fig)

print("\nSaved results to:", OUTPUT_DIR)
print(" - state_year_weather_yield.csv")
print(" - weather_yield_correlations.csv")
print(" - correlation_heatmap.png")
print(" - Scatterplots for each weather feature")

print("\nReminder: correlation does not prove causation.")
print("State, crop mix, irrigation, soil, and farming practices can also affect yield.")
