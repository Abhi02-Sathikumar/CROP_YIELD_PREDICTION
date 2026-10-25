# CROP YIELD PREDICTION & SMART IRRIGATION ADVISORY FROM WEATHER + SOIL DATA

## PROJECT REPORT

**Project Name:** AgriPulse

**Project Title:** Crop Yield Prediction & Smart Irrigation Advisory from Weather + Soil Data

**Submitted By:**  
[Team Member 1 Name]  
[Team Member 2 Name]  
[Team Member 3 Name]  
[Add all team members]

**Team Guide / Mentor:** [Guide's Name]

**Department / Institution:** [Institution Name]

**Submission Date:** 31 October 2026

---

# 1. TEAM MEMBERS AND DUTIES

| Team Member | Responsibilities |
|---|---|
| [Member 1] | Data collection, dataset understanding and preprocessing |
| [Member 2] | Exploratory Data Analysis and visualization |
| [Member 3] | Machine Learning model training and evaluation |
| [Member 4] | Streamlit dashboard development, testing and documentation |
| [Member 5] | ....................................................|

*Note: Replace the sample responsibilities with the actual work completed by each team member.*

---

# 2. PROJECT OVERVIEW

## 2.1 Introduction

Agriculture depends on several factors, including weather conditions, soil properties, crop selection and water availability. Inaccurate crop yield estimates and improper irrigation decisions can affect agricultural productivity and increase costs.

The project, **Crop Yield Prediction & Smart Irrigation Advisory from Weather + Soil Data**, aims to develop a data-driven agricultural analytics application using historical crop production records, weather information, soil nutrient samples and Machine Learning techniques.

The application, named AgriPulse, provides an interactive dashboard to analyze historical agricultural data, explore weather–yield relationships, visualize regional production patterns and estimate crop yield.

It also provides a preliminary irrigation advisory using historical weather observations.

## 2.2 Problem Statement

Farmers and agricultural planners need useful information about crop productivity and weather conditions to support agricultural planning.

Traditional analysis may require considerable effort to compare historical production records, weather patterns and regional differences.

This project addresses the problem by combining data analysis, visualization and Machine Learning in a single interactive application.

## 2.3 Objectives

1. Analyze historical crop production and yield patterns.
2. Study the relationship between weather variables and crop yield.
3. Develop and compare Machine Learning regression models.
4. Provide preliminary irrigation-related advisories using historical weather data.
5. Visualize state-level crop production and yield.
6. Develop an interactive Streamlit dashboard.
7. Generate downloadable analytical reports.

---

# 3. DATASETS AND DATA SOURCES

The project uses the following datasets.

## 3.1 Crop Production Dataset

The crop production dataset contains historical agricultural records.

Important columns include:

- State_Name
- District_Name
- Crop_Year
- Season
- Crop
- Area
- Production

These records support historical production analysis and yield calculation.

## 3.2 Crop Recommendation Dataset

The crop recommendation dataset contains sample information about soil nutrients and environmental conditions.

Important columns include:

- N — Nitrogen
- P — Phosphorus
- K — Potassium
- Temperature
- Humidity
- pH
- Rainfall
- Crop label

This dataset supports exploratory soil nutrient analysis.

The soil sample records are not directly linked to the historical crop production records by district and year. Therefore, they are analyzed separately.

## 3.3 Historical Weather Dataset

Historical weather data is obtained through the Open-Meteo Historical Weather API.

Weather variables include:

- Mean temperature
- Average maximum temperature
- Average minimum temperature
- Total precipitation
- Maximum wind speed

The available state-level yearly weather records cover 2000–2014.

These data are used for historical weather analysis and for constructing weather-related features.

## 3.4 Data Preprocessing

The raw datasets are examined and prepared for analysis.

The preprocessing workflow includes:

1. Loading the datasets.
2. Inspecting rows, columns and data types.
3. Standardizing column names and text fields.
4. Checking missing values and duplicate records.
5. Validating area and production values.
6. Calculating crop yield.
7. Preparing weather variables.
8. Merging compatible crop and weather records.
9. Saving processed datasets for further analysis.

Crop yield is calculated as:

**Yield = Production / Area**

The calculated value follows the units of the source production and area fields.

---

# 4. METHODOLOGY

The project follows a data science workflow.

## Step 1: Data Collection

Historical crop production data, soil nutrient samples and historical weather information are collected from the available datasets and weather API.

## Step 2: Data Preprocessing

The datasets are inspected, cleaned and standardized. Invalid area values and missing production records are handled during crop production preprocessing.

## Step 3: Data Integration

Crop production records are combined with compatible historical weather records using geographical and yearly identifiers.

## Step 4: Exploratory Data Analysis

Statistical summaries and visualizations are used to study production trends, crop distribution, regional differences and weather–yield relationships.

## Step 5: Feature Engineering

Features such as crop, state, district, season, year, cultivated area and available weather measurements are prepared for Machine Learning.

Categorical variables are encoded, and missing feature values are handled using preprocessing pipelines where appropriate.

## Step 6: Model Training

Regression models, including Random Forest Regressor and XGBoost Regressor, are trained and compared.

## Step 7: Model Evaluation

Model performance is assessed using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

## Step 8: Visualization and Application Development

An interactive dashboard is developed using Streamlit and Plotly.

The dashboard allows users to filter data, explore visualizations, inspect yield estimates and download analytical results.

## Step 9: Report Generation

The application provides downloadable CSV data and a PDF summary of filtered historical crop production records.

---

# 5. TECHNOLOGIES, LIBRARIES AND PACKAGES

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Data preprocessing, regression models and evaluation |
| XGBoost | Gradient boosting regression |
| Joblib | Saving and loading trained models |
| Plotly | Interactive graphs and maps |
| Streamlit | Web application development |
| ReportLab | PDF report generation |
| Open-Meteo Historical Weather API | Historical weather data collection |
| GeoJSON | Geographical boundary information |
| Git and GitHub | Version control and project repository management |
| SHAP | Explaining Machine Learning model predictions |

---

# 6. IMPLEMENTED FEATURES

## Feature 1: Crop Yield Prediction

This feature estimates crop yield using a trained Machine Learning model.

The application allows users to select crop-related and geographical information, season, historical weather year and cultivated area.

The selected inputs and corresponding historical weather features are passed to the weather-aware Random Forest model.

The application displays the estimated yield.

The result is an estimate rather than a guaranteed agricultural outcome. Model validation and data limitations must be considered when interpreting predictions.

## Feature 2: Smart Irrigation Advisory

This feature examines historical weather observations to provide preliminary irrigation-related information.

The application compares selected historical rainfall and temperature observations with historical reference values for the selected state.

It displays messages indicating whether the observed rainfall is relatively low or the temperature is relatively high compared with historical observations.

The current implementation is a demonstration based on historical weather data. It does not provide a real-time weather forecast or a validated irrigation water-depth recommendation.

## Feature 3: Weather–Yield Analytics

This feature analyzes the relationship between historical weather variables and crop yield.

Users can explore weather measurements, scatter plots and correlation values.

The analysis helps identify statistical associations between weather conditions and agricultural productivity.

Correlation does not establish causation, and the results may be influenced by other factors, including crop selection, soil conditions, irrigation and farming practices.

## Feature 4: Regional Production Mapping

This feature visualizes historical crop production and yield across Indian states and union territories.

The application uses geographical boundary information and Plotly to create an interactive choropleth map.

Users can examine regional production patterns and compare mapped production or yield values.

The current map uses state/territory-level boundaries rather than district-level geographical boundaries.

## Additional Features

### Soil Nutrient Analysis

The dashboard displays the distribution of nitrogen, phosphorus and potassium values in the available crop recommendation sample dataset.

### Explainable AI

SHAP is used in separate analysis scripts to examine feature contributions to trained model predictions.

### Downloadable Reports

Users can download filtered crop production records as CSV files and generate PDF summaries containing production statistics and leading crops and regions.

---

# 7. PROJECT RESULTS AND EVALUATION

The saved baseline model comparison contains the following results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost | 11.72 | 254.08 | 0.881 |
| Random Forest | 12.14 | 321.52 | 0.809 |

In this saved comparison, XGBoost has lower MAE and RMSE and a higher R² than Random Forest.

The separate weather-aware model comparison shows lower R² values of approximately 0.08 for both models.

This indicates that the weather-aware models require further investigation and validation before their predictions can be considered reliable for practical agricultural decisions.

The evaluation results depend on the dataset, target variable, preprocessing and validation methodology.

---

# 8. LIMITATIONS

1. The available weather observations are historical and state-level rather than real-time, district-specific forecasts.
2. The soil nutrient sample dataset is not directly linked to historical crop production by district and year.
3. The current irrigation advisory is a preliminary historical-weather demonstration rather than a validated irrigation recommendation system.
4. The regional map uses state/territory-level geographical boundaries.
5. Weather-aware model evaluation results indicate that further validation and improvement are needed.
6. Predictions should not be treated as guaranteed crop yields or used as the sole basis for agricultural decisions.

---

# 9. CONCLUSION

The Crop Yield Prediction & Smart Irrigation Advisory project demonstrates how Data Science, Machine Learning and interactive visualization can be applied to agricultural data.

The project combines historical crop production records, weather information and soil nutrient samples to explore agricultural productivity patterns.

The Streamlit application provides an interface for historical production analysis, weather–yield analytics, regional mapping, soil nutrient exploration, yield estimation and preliminary irrigation advisory.

The project also demonstrates the importance of model evaluation, data quality and explainability in Machine Learning applications.

Further development could include district-level weather integration, real-time forecast data, validated crop water requirement calculations, improved model validation and stronger predictive performance.

---

# 10. PROJECT LINKS AND REFERENCES

## Project Repository

GitHub Repository: [Insert the final repository link]

## Application Link

Deployed Application: [Insert the live application link, if available]

## Screen Recording

Cloud Recording Link: [Insert the uploaded screen-recording link]

## References

1. Open-Meteo Historical Weather API documentation.
2. Scikit-learn documentation.
3. XGBoost documentation.
4. Streamlit documentation.
5. SHAP documentation.
6. Plotly documentation.
7. ReportLab documentation.
8. Crop production and crop recommendation datasets used in the project.

**Note:** Add the exact dataset URLs and documentation links used by the team before final submission.
