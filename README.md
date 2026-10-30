# AgriPulse: Crop Yield Prediction and Irrigation Advisory

AgriPulse is a Streamlit dashboard for exploring Indian crop-production records, comparing historical weather and yield, viewing state-level production patterns, inspecting crop-recommendation soil samples, estimating crop yield, and generating a weather-based irrigation advisory.

The app is a decision-support and educational tool. Its yield predictions and irrigation depths are estimates, not guarantees or agronomic prescriptions.

## Contents

- [Features](#features)
- [How the app works](#how-the-app-works)
- [Project structure](#project-structure)
- [Data and model files](#data-and-model-files)
- [Install and run](#install-and-run)
- [Using the dashboard](#using-the-dashboard)
- [Training and analysis scripts](#training-and-analysis-scripts)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Troubleshooting](#troubleshooting)
- [Technology](#technology)

## Features

- **Production overview:** summarizes production, cultivated area, and weighted yield for selected crop, state, district, season, and year filters.
- **Weather analytics:** compares historical crop yields with matched weather observations, with 2D and 3D charts and descriptive correlations.
- **Regional map:** maps production/yield statistics using the bundled India state boundaries.
- **Soil nutrients:** explores N, P, and K values in the crop-recommendation sample dataset.
- **Irrigation advisory:** retrieves a 3-, 5-, or 7-day forecast for an Indian location and estimates a daily crop-water balance. The manual calculator is prefilled with forecast-period daily averages when available and provides a descriptive one-day water-balance summary.
- **Yield prediction:** uses the bundled weather-aware Random Forest model to estimate yield and production for selected crop, state, season, district, area, and year.
- **Reports:** downloads the filtered production records as CSV and a summary of the selected records as PDF.
- **Display mode:** switch between light and dark dashboard themes.

## How the app works

### Daily irrigation advisory

1. Enter a city, district, or town in India, then choose a crop, crop growth stage, and forecast horizon.
2. The app resolves the location using the Open-Meteo Geocoding API and requests daily maximum temperature, mean relative humidity, precipitation, and FAO reference evapotranspiration (ET₀) from the Open-Meteo Forecast API.
3. For each forecast day, it estimates crop water use as:

   `crop water use = ET₀ × crop Kc × growth-stage multiplier`

4. It estimates effective rainfall as 80% of forecast precipitation, subtracts that from crop water use, and divides the resulting net need by an assumed irrigation efficiency of 85% to estimate gross irrigation depth.
5. The results table shows the daily forecast, estimated water use and irrigation depths, weather-stress indicator, and an action message.

Forecast data is cached for 30 minutes. **Use manual weather inputs instead** accepts ET₀, rainfall, temperature, and humidity for a one-day estimate. After a successful forecast, these inputs are prefilled with the average daily temperature, humidity, rainfall, and ET₀ over the returned forecast days; each value can be edited before calculating. The manual result describes crop water use, effective rainfall, net water need, estimated gross irrigation, and the weather-only stress indicator.

The crop coefficients and stage multipliers are illustrative defaults in `src/irrigation.py`; they are not calibrated for a particular farm, soil, irrigation system, or local crop calendar. The forecast advisory calculates each day independently; it does not carry soil water balance forward between forecast days. Always check field soil moisture and local agronomic guidance before irrigating.

### Crop yield prediction

The app loads `models/random_forest_weather.joblib` and uses the artifact's declared feature names to prepare a prediction. It pairs the selected state with its latest available historical state/year weather record, then estimates yield and multiplies that yield by the converted cultivated area to estimate production.

The bundled model and weather observations represent historical data through 2014. Choosing a prediction year outside 2000–2014 is an extrapolation, not a current or live forecast. The app does not generate a confidence interval or an explanation for an individual prediction.

## Project structure

```text
.
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── assets/
│   ├── india_state.geojson         # State boundaries for the map
│   ├── logo.png
│   ├── maize_corn_plant.glb        # Overview background model
│   └── rice_plant.glb
├── data/
│   ├── raw/                        # Source data, if supplied separately
│   └── processed_data/             # Dashboard input CSVs
├── models/                         # Trained model artifacts
├── notebooks/                      # Exploratory and project notebooks
├── outputs/                        # Generated analysis/model outputs
├── src/
│   ├── explain.py                  # Model explanation workflow
│   ├── explain_weather.py          # Weather model explanation workflow
│   ├── irrigation.py               # Water balance and weather-stress estimates
│   ├── merge_weather_crop.py       # Crop/weather data preparation
│   ├── predict.py                  # Prediction helper for training artifacts
│   ├── preprocess.py               # Data preparation helpers
│   ├── train.py                    # Model training workflow
│   ├── train_weather.py            # Weather-aware model training workflow
│   ├── weather_correlation.py      # Weather/yield analysis
│   └── weather_forecast.py         # Open-Meteo geocoding and forecast client
└── tests/                          # Test directory
```

The project also includes a `venv/` folder in some local copies. It is an environment-specific directory and should not be copied or committed; create your own virtual environment instead.

## Data and model files

The dashboard expects the following processed files under `data/processed_data/`:

| File | Purpose | Required columns |
|---|---|---|
| `crop_production_preprocessed.csv` | Main production records; required for app startup | `State`, `District`, `Year`, `Season`, `Crop`, `Area`, `Production`; `Yield` is optional and is derived as `Production / Area` when absent |
| `crop_weather_merged.csv` | Matched crop/weather records for weather analytics | Production columns plus `Yield`, `Mean_Temperature`, `Avg_Max_Temp`, `Avg_Min_Temp`, `Total_Precipitation_mm`, `Max_Wind_Speed` |
| `india_states_yearly_weather_preprocessed.csv` | State/year weather context and yield-model inputs | `State`, `Year`, and the five weather columns above |
| `crop_recommendation_preprocessed.csv` | Soil-sample exploration | `N`, `P`, `K`, `label`; `temperature`, `humidity`, `ph`, and `rainfall` are used when present |

Other related files, such as `weather_preprocessed.csv`, `crop_weather_merge_all.csv`, and `india_states_yearly_weather_2000_2014.csv`, may be intermediate or analysis inputs. The Streamlit dashboard uses the exact filenames in the table above.

The yield page uses `models/random_forest_weather.joblib`. Other model artifacts in `models/` are retained for training and comparison workflows; they are not automatically selected by the dashboard. The regional map uses `assets/india_state.geojson`.

### Units and data interpretation

- Cultivated area is stored/displayed in **hectares**; the yield page also accepts acres and converts them to hectares.
- Production is in **metric tonnes**.
- Yield is in **metric tonnes per hectare**.
- Crop/weather data and state/year weather observations are historical. Repeated state/year weather values across crop records do not represent independent daily observations.
- Soil recommendation samples are not linked to the production rows. Their nutrient distributions must not be interpreted as measured nutrient-to-yield relationships.

Use source datasets with appropriate permissions and retain their attribution and license information. Do not treat different datasets as linked unless the location, time period, and measurement definitions genuinely match.

## Install and run

### Prerequisites

- Python 3.10–3.12 recommended for compatibility with the pinned scikit-learn dependency.
- Internet access for the live irrigation forecast.
- The processed data, map, logo, and model files described above.

### Windows PowerShell

From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell prevents environment activation, invoke the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) when the app starts. Stop the local server with **Ctrl+C** in the terminal.

## Using the dashboard

1. Start the app using the instructions above.
2. Choose a dashboard page from **Workspace** in the sidebar.
3. Use the sidebar filters for crop, state, district, season, and crop year where applicable.
4. On **Irrigation advisory**, enter a location and submit the forecast form. The page needs an internet connection to contact Open-Meteo. If that is unavailable, expand the manual-input option.
5. On **Yield prediction**, select the prediction inputs and submit the form. Review the displayed historical-weather year and any extrapolation warning.
6. On **Reports**, download the current filtered records as CSV or the production summary as PDF.

The dashboard may show a clear warning or omit a weather/soil feature if an optional processed dataset is missing. The production CSV and required visual assets must be present for startup.

## Training and analysis scripts

The repository includes notebooks and Python scripts for data preparation, weather joins, model training, predictions, explanations, and weather/yield analysis. The current dashboard loads prebuilt artifacts and processed CSVs; it does not run a training job automatically.

The main weather-aware training script is `src/train_weather.py`. It reads `data/processed_data/crop_weather_merged.csv`, trains Random Forest and XGBoost regressors, evaluates them, and writes model/output artifacts. The target (`Yield`) is trained on a `log1p` scale and transformed back to yield units for predictions, which reduces the influence of the dataset's highly skewed yield values. Production is excluded from model features because it is used to calculate the target. Review the input paths and script settings before running it:

```powershell
.\.venv\Scripts\python.exe src\train_weather.py
```

```bash
python src/train_weather.py
```

Training requires the matching input dataset and XGBoost. Training or replacing a model can change prediction behavior; validate it on a suitable holdout set before using it in the dashboard.

## Limitations and responsible use

- This dashboard is not an official agricultural service and is not a substitute for local extension or agronomy advice.
- Forecast availability and accuracy depend on Open-Meteo and the selected location. Forecast values can change.
- Irrigation coefficients, effective-rain fraction, irrigation efficiency, and stage multipliers are general illustrative assumptions. The advisory has no field soil sensor input in the forecast workflow.
- Weather stress is an atmospheric-demand screening proxy based on temperature and humidity, not a measured Crop Water Stress Index (CWSI). Physiological CWSI requires canopy-temperature measurements and wet/dry reference baselines.
- Daily irrigation estimates are independent and do not account for soil texture, rooting depth, field capacity, irrigation timing, runoff, drainage, or a changing soil-water balance.
- Yield-model inputs rely on historical records ending in 2014, and predictions for newer years extrapolate beyond that training period. The dashboard does not display uncertainty intervals.
- Weather/yield visualizations show association, not causation. The soil sample dataset is not matched to production records.
- Results depend on input data quality, coverage, definitions, and units. Review data provenance and validate predictions for the intended region and crop before operational use.

## Troubleshooting

| Symptom | What to check |
|---|---|
| App says the production dataset is missing | Confirm `data/processed_data/crop_production_preprocessed.csv` exists and is readable. |
| A weather, soil, or map section is unavailable | Check that its optional CSV or `assets/india_state.geojson` exists and includes the required columns/identifiers. |
| Yield model unavailable | Confirm `models/random_forest_weather.joblib` exists and can be loaded in the active environment. The model was serialized with a particular scikit-learn version; use the pinned version in `requirements.txt`. |
| Location cannot be found | Try a more specific Indian city or district name. The forecast geocoder is restricted to India. |
| Forecast request fails | Check internet access and try again. Alternatively, use the manual single-day advisory. |
| Dependency installation fails | Check that the selected Python version is compatible with the pinned dependencies, then recreate the virtual environment and reinstall. |

## Technology

- Python, pandas, NumPy
- Streamlit
- Plotly
- scikit-learn, XGBoost, joblib
- ReportLab
- Open-Meteo Geocoding and Forecast APIs
- GeoJSON state boundaries
