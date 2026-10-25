# Crop Yield Prediction & Smart Irrigation Advisory from Weather + Soil Data

## 1. Project Introduction

### Project Name
Crop Yield Prediction & Smart Irrigation Advisory from Weather + Soil Data

### Project Overview
Agriculture depends on several factors, including weather conditions, soil nutrients, rainfall, temperature, humidity, and crop type.

Our project uses historical crop production and weather data to analyse agricultural patterns, predict crop yield, and provide an irrigation advisory.

The project includes a dashboard that helps users explore agricultural data through charts, maps, predictions, and reports.

### Simple Explanation
Farmers need to understand how much crop they may produce and whether weather and soil conditions are suitable for crop growth.

Our project uses data analysis and machine learning to support these decisions.

**Malayalam explanation:** Crop production-ne affect cheyyunna weather, soil, crop type ennavayude data use cheythu yield predict cheyyukayum irrigation advisory provide cheyyukayum cheyyunna project aanu ithu.

---

## 2. Main Objectives

1. Analyse historical crop production data.
2. Study crop production patterns across states, districts, and years.
3. Understand the relationship between weather conditions and crop yield.
4. Build machine learning models for crop yield prediction.
5. Provide a smart irrigation advisory using available weather and soil-related information.
6. Display the results through an interactive dashboard.
7. Help users understand agricultural data more easily.

---

## 3. Project Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Random Forest | Crop yield prediction |
| XGBoost | Alternative machine learning model |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive charts and visualisations |
| Joblib | Saving and loading trained models |
| GeoJSON | Geographic boundaries for the regional map |
| Open-Meteo API | Historical weather data retrieval in the weather workflow |
| GitHub | Project code storage and version control |

---

## 4. Project Workflow

The general workflow is:

1. Collect the datasets.
2. Understand the datasets.
3. Clean and preprocess the data.
4. Analyse crop production and weather patterns.
5. Combine relevant crop and weather information.
6. Prepare features for machine learning.
7. Train and evaluate machine learning models.
8. Save the trained models.
9. Load the model and prepared data into the Streamlit application.
10. Display predictions, charts, maps, and advisory information.
11. Test the application and prepare the project report.

### Why do we follow these steps?

Raw data may contain missing values, duplicates, inconsistent formats, and information that is not directly suitable for machine learning.

Preprocessing prepares the data. Analysis helps us understand patterns. Machine learning identifies relationships in the training data and makes predictions. The dashboard presents the results in an understandable format.

---

# 5. Dataset Explanation

## 5.1 Crop Production Dataset

File: `crop_production.csv`

This dataset contains historical agricultural production information.

Important columns:

| Column | Meaning |
|---|---|
| State_Name | Name of the state |
| District_Name | Name of the district |
| Crop_Year | Year of crop production |
| Season | Agricultural season |
| Crop | Name of the crop |
| Area | Land area used for cultivation |
| Production | Total crop production |

### What is this dataset used for?

It is used to study historical crop production, compare crops and regions, and prepare the target variable for yield prediction.

### What is crop yield?

Crop yield represents the amount of crop produced per unit of cultivated land.

A common calculation is:

**Yield = Production / Area**

The exact unit depends on the units used in the dataset.

## 5.2 Crop Recommendation Dataset

File: `Crop_recommendation.csv`

This dataset contains soil nutrient and environmental measurements associated with crop labels.

Important columns include:

- N — Nitrogen level
- P — Phosphorus level
- K — Potassium level
- Temperature
- Humidity
- pH
- Rainfall
- Label — Crop label

### What is it used for?

It provides soil and environmental information that can support crop-related analysis.

**Important:** Crop recommendation and crop yield prediction are different tasks. A crop recommendation dataset should not automatically be described as a direct yield-prediction dataset.

## 5.3 Historical Weather Dataset

File: `india_states_yearly_weather_2000_2014.csv`

This dataset contains historical weather information at the state-year level for the period 2000–2014.

Weather variables can include temperature, rainfall, and humidity, depending on the available columns.

### What is it used for?

It helps analyse historical weather patterns and investigate how weather conditions relate to crop production.

## 5.4 Processed Datasets

The project also contains processed datasets, including:

- `crop_production_preprocessed.csv`
- `crop_weather_merged.csv`
- `crop_weather_merge_all.csv`
- `weather_preprocessed.csv`
- `india_states_yearly_weather_preprocessed.csv`
- `training_data.csv`

These files contain data prepared for analysis, merging, or model training.

The exact columns and row counts should be verified in the actual files before the presentation.

---

# 6. Data Preprocessing

## What is data preprocessing?

Data preprocessing is the process of preparing raw data for analysis and machine learning.

### Main activities

1. Inspect the dataset.
2. Check column names and data types.
3. Identify missing values.
4. Check duplicate records.
5. Correct inconsistent data formats.
6. Handle invalid or unsuitable values.
7. Prepare the target variable.
8. Select relevant features.
9. Merge datasets when appropriate.
10. Save the processed dataset.

### Why is preprocessing important?

Poor-quality data can affect analysis and model performance. Preprocessing makes the data more consistent and suitable for the next stages.

### Viva question

**Q: Why did you perform data preprocessing?**

Answer:

We performed data preprocessing to check data quality, handle missing or duplicate records where necessary, prepare useful features, and make the dataset suitable for analysis and machine learning.

---

# 7. Data Merging

File: `src/merge_weather_crop.py`

### What is data merging?

Data merging means combining information from different datasets using common columns or keys.

For example, crop production data may contain state and year information, while weather data contains state and year information.

These fields can be used as possible matching keys.

### Why merge crop and weather data?

It helps us analyse agricultural production alongside historical weather conditions.

### Important limitation

The weather data is at the state-year level, whereas crop data may be at the district, crop, and season level. A merge must use appropriate keys and be checked carefully to avoid incorrect matches or duplicated weather records.

---

# 8. Exploratory Data Analysis (EDA)

EDA means Exploratory Data Analysis.

It involves examining data through summary statistics, charts, and comparisons before building a machine learning model.

### Examples of analysis

- Crop production by year.
- Production by state or district.
- Comparison of different crops.
- Weather trends over time.
- Relationship between weather variables and yield.
- Regional differences in agricultural production.

### Why is EDA important?

EDA helps us understand the data, identify patterns, detect unusual values, and decide which features may be useful for modelling.

### Viva question

**Q: What is EDA?**

Answer:

EDA stands for Exploratory Data Analysis. It is the process of understanding a dataset using statistics, tables, and visualisations before building a model.

---

# 9. Machine Learning Models

The project includes Random Forest and XGBoost model files.

## 9.1 Random Forest

Random Forest is an ensemble machine learning algorithm that combines predictions from multiple decision trees.

For regression, the trees' predictions are combined to produce a numerical prediction.

### Advantages

- Can learn nonlinear relationships.
- Can work with several input features.
- Combines multiple decision trees.
- Can be used for regression problems.

### Simple viva answer

Random Forest is a machine learning algorithm that combines multiple decision trees to make a prediction.

## 9.2 XGBoost

XGBoost is a boosting algorithm that builds trees sequentially, with each new stage attempting to improve the model.

### Advantages

- Can model complex relationships.
- Often performs well on structured datasets.
- Supports regression and classification tasks.

### Simple viva answer

XGBoost is a boosting algorithm that builds decision trees sequentially to improve predictive performance.

## 9.3 Features and Target

**Features** are the input variables used by the model.

Possible features in an agricultural prediction task include crop type, cultivated area, location, year, season, and available weather measurements.

**Target** is the value that the model attempts to predict.

For a yield-prediction task, the target is the yield value.

The exact feature list used by each saved model should be checked in the training code or model metadata.

---

# 10. Model Evaluation

Model evaluation helps us understand how accurately a machine learning model predicts unseen data.

## 10.1 MAE — Mean Absolute Error

MAE measures the average absolute difference between actual and predicted values.

A lower MAE generally indicates smaller prediction errors.

## 10.2 RMSE — Root Mean Squared Error

RMSE measures prediction errors while giving larger errors more influence.

A lower RMSE generally indicates better performance, provided the models are evaluated on the same target, units, and test data.

## 10.3 R² Score

R² measures how well the model explains variation in the target relative to a baseline.

A value closer to 1 can indicate better performance. A low or negative score can indicate poor predictive performance.

### Saved baseline comparison

The repository contains a model comparison file reporting these values:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost | 11.72 | 254.08 | 0.881 |
| Random Forest | 12.14 | 321.52 | 0.809 |

**Important:** These are values recorded in the saved baseline comparison. Before presenting them as final results, verify the target units, evaluation dataset, and training code.

The repository also contains a separate weather-aware model comparison. Do not mix the two comparisons because their reported scores differ substantially.

### Viva question

**Q: Which evaluation metrics did you use?**

Answer:

We use MAE, RMSE, and R² to evaluate regression models. MAE and RMSE measure prediction errors, while R² indicates how well the model explains variation in the target.

---

# 11. Saved Model Files

The `models/` directory contains saved machine learning models and supporting files.

Examples include:

- `random_forest.joblib`
- `xgboost.joblib`
- `random_forest_weather.joblib`
- `xgboost_weather.joblib`
- `best_model.pkl`
- `features.pkl`
- `model_metadata.pkl`
- `model_comparison.csv`

### What is a saved model?

After training, a model can be saved to a file. The application can load that file and use the trained model for prediction without training it again every time.

### What is Joblib?

Joblib is a Python library commonly used to save and load Python objects, including trained machine learning models.

**Important:** The application should load the model that matches the features and preprocessing used during its training. The presence of a model file in the repository does not necessarily mean the dashboard uses that file.

---

# 12. Smart Irrigation Advisory

### What is smart irrigation?

Smart irrigation aims to support decisions about when or how much to water crops by considering relevant environmental and soil conditions.

Possible factors include:

- Rainfall
- Temperature
- Humidity
- Soil moisture, if available
- Soil characteristics
- Crop requirements

### How does this project help?

The project provides an irrigation advisory feature using available weather and soil-related information.

### Important limitation

The current advisory should be presented as a demonstration or preliminary screening tool unless its recommendations have been validated against suitable agronomic data.

Historical weather information is not the same as a live weather forecast. Do not claim that the system provides real-time weather predictions unless that functionality is actually implemented.

### Viva question

**Q: Why did you include an irrigation advisory?**

Answer:

We included an irrigation advisory to demonstrate how weather and soil-related information can support agricultural water-management decisions. The advisory needs further validation before it can be used as a reliable field recommendation.

---

# 13. Streamlit Dashboard

Main file: `app.py`

### What is Streamlit?

Streamlit is a Python framework used to build interactive web applications, especially for data science and machine learning projects.

### Dashboard sections

The application includes these sections:

1. Overview
2. Weather Analytics
3. Regional Map
4. Soil Nutrients
5. Irrigation Advisory
6. Yield Prediction
7. Reports

### What does each section do?

**Overview:** Provides a summary of the project and important agricultural information.

**Weather Analytics:** Displays weather-related analysis and visualisations.

**Regional Map:** Helps users explore regional agricultural information using geographic visualisation.

**Soil Nutrients:** Displays information about soil nutrients and related measurements available in the project.

**Irrigation Advisory:** Presents the implemented irrigation-related advisory.

**Yield Prediction:** Provides the yield-prediction interface.

**Reports:** Supports viewing or downloading available reports, depending on the implemented application functions.

### Viva question

**Q: Why did you use Streamlit?**

Answer:

We used Streamlit because it allows us to build an interactive dashboard using Python without developing a separate frontend from scratch.

---

# 14. Explanation of Important Project Files

## Main files

### `app.py`
The main Streamlit application. It connects the dashboard interface with the project's data, charts, prediction functions, and advisory features.

### `README.md`
Explains the project, setup instructions, technologies, and how to run the application.

### `requirements.txt`
Lists the Python packages required by the project.

### `.gitignore`
Lists files or folders that Git should normally ignore, such as certain temporary files and local environment files.

## Source-code files

### `src/preprocess.py`
Contains preprocessing logic. Check the code to confirm the exact cleaning and transformation operations.

### `src/merge_weather_crop.py`
Contains logic for combining crop production and weather information.

### `src/train.py`
Contains a model-training workflow. Verify the exact model and features used by reading the code.

### `src/train_weather.py`
Contains a weather-related model-training workflow.

### `src/predict.py`
Contains prediction-related functions.

### `src/irrigation.py`
Contains irrigation-advisory logic.

### `src/weather_correlation.py`
Contains analysis of relationships between weather and agricultural variables.

### `src/explain.py`
Contains model-explanation functionality.

### `src/explain_weather.py`
Contains explanation functionality for the weather-related modelling workflow.

## Data files

### `data/raw/`
Contains original datasets.

### `data/processed_data/`
Contains datasets prepared for analysis or modelling.

## Model files

### `models/`
Contains saved models, feature information, metadata, and model comparison results.

## Notebooks

### `notebooks/`
Contains Jupyter notebooks used for experimentation, analysis, or data preparation.

### `Weather_data_fetch_workflow.ipynb`
Contains a weather-data retrieval workflow using the Open-Meteo Historical Weather API.

## Assets

### `assets/india_state.geojson`
Contains geographic boundary data used for regional mapping.

## Outputs

### `outputs/`
Contains generated results such as plots, model comparisons, correlation results, or model-explanation outputs.

---

# 15. How to Run the Project

The exact command may depend on the project setup and installed dependencies.

A typical setup is:

1. Download or clone the repository.
2. Open the project folder.
3. Create and activate a Python virtual environment if required.
4. Install the packages listed in `requirements.txt`.
5. Run the Streamlit application.

Example commands:

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

After the application starts, open the local address shown in the terminal.

If an error appears, check the Python environment, installed packages, file paths, and model-file availability.

---

# 16. Project Task Time Estimation

The following table is a planning estimate only. Replace these ranges with the team's actual recorded time before presenting them as completed-work durations.

| Task | Estimated time |
|---|---|
| Project planning | 2–4 hours |
| Dataset collection | 4–8 hours |
| Data understanding | 2–4 hours |
| Data preprocessing | 4–8 hours |
| Data merging | 3–6 hours |
| Exploratory data analysis | 4–8 hours |
| Model training | 4–8 hours |
| Model evaluation | 2–4 hours |
| Irrigation advisory | 4–8 hours |
| Dashboard development | 6–12 hours |
| Testing and debugging | 3–6 hours |
| Report preparation | 3–5 hours |

### How to answer the teacher

**Question: How long did you take for data preprocessing?**

Answer using the actual time spent by your team. Explain that preprocessing involved inspecting the data, checking its quality, and preparing it for analysis.

**Question: How long did you take to develop the dashboard?**

State the actual time spent developing and testing the dashboard. Explain that the work included creating the interface, displaying charts, and connecting the available project features.

Do not present estimated times as exact historical records.

---

# 17. Limitations of the Project

1. Historical data may not represent current agricultural conditions.
2. Weather data at state-year level may not capture local district-level weather differences.
3. The soil recommendation dataset may not be directly linked to the crop-production records.
4. Model performance depends on the quality and representativeness of the training data.
5. The irrigation advisory requires agricultural validation before practical field use.
6. Yield predictions are estimates, not guarantees of actual harvest.
7. The project should not be described as providing live weather information unless a live-data feature is implemented.

---

# 18. Common Viva Questions and Answers

### Q1. What is your project about?

Our project analyses historical crop production, weather, and soil-related data to support crop yield prediction and irrigation advisory through an interactive dashboard.

### Q2. What is the main purpose of the project?

The main purpose is to use agricultural data analysis and machine learning to support better-informed agricultural decisions.

### Q3. What is machine learning?

Machine learning is a branch of artificial intelligence in which computers learn patterns from data to make predictions or decisions.

### Q4. What is regression?

Regression is a machine learning task used to predict numerical values, such as crop yield.

### Q5. Why did you use Random Forest?

Random Forest combines multiple decision trees and can learn complex relationships in structured data.

### Q6. What is XGBoost?

XGBoost is a boosting algorithm that builds trees sequentially to improve predictions.

### Q7. What is the target variable?

For the yield-prediction task, the target variable is crop yield. The exact target column should be verified in the training code.

### Q8. Why is data preprocessing important?

It helps prepare the data for analysis and modelling by handling data-quality problems and creating suitable features.

### Q9. What is a feature?

A feature is an input variable used by a machine learning model to make a prediction.

### Q10. What is model evaluation?

Model evaluation measures how well a model performs on evaluation data.

### Q11. What is the purpose of Streamlit?

Streamlit is used to create an interactive dashboard using Python.

### Q12. What is the difference between production and yield?

Production is the total quantity harvested. Yield is production per unit of cultivated area.

### Q13. Does the application provide guaranteed predictions?

No. Predictions are estimates based on the model and available data. Actual yield depends on several factors.

### Q14. Does the irrigation advisory guarantee the correct amount of water?

No. The advisory is a demonstration based on available information and needs validation using suitable agricultural data before practical use.

### Q15. What are the limitations of the project?

Important limitations include historical data, differences in geographical data resolution, dataset compatibility, and the need for further validation.

---

# 19. Final Presentation Checklist

Before the presentation:

- Understand the project objective.
- Open the application and test every dashboard section.
- Understand the datasets and important columns.
- Read the main parts of `app.py`.
- Understand the purpose of each source file.
- Identify which model is actually loaded for prediction.
- Verify the reported evaluation metrics.
- Prepare an honest explanation of the irrigation advisory.
- Confirm the actual tasks completed by each team member.
- Record actual time spent wherever available.
- Avoid claiming features that are not implemented.
- Keep the project report and presentation consistent with the code.

## Final Summary

This project combines agricultural data analysis, machine learning, weather-related analysis, and an interactive dashboard. It demonstrates how data-driven tools can support crop yield estimation and preliminary irrigation decision-making.

The results should be interpreted in the context of the available datasets, model evaluation, and project limitations.
