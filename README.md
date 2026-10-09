🌾 Crop Yield Prediction & Smart Irrigation Advisory

An AI-powered agricultural intelligence system that predicts crop yield and provides smart irrigation recommendations using weather, soil, and historical crop production data.

📌 Project Overview

Agriculture depends heavily on weather conditions, soil fertility, and effective water management. Inaccurate crop yield estimates can lead to financial losses, while improper irrigation can waste water or negatively affect crop growth.

This project combines Data Science, Data Analytics, and Machine Learning to help farmers and agricultural planners make data-driven decisions.

The system analyzes historical crop production records, weather conditions, and soil nutrient information to:

Predict seasonal crop yield in tons per hectare.
Analyze the relationship between weather conditions and crop productivity.
Identify regional crop production patterns across Indian districts.
Generate daily irrigation advisories based on weather forecasts and crop water requirements.
Explain yield predictions using SHAP feature importance.
Visualize agricultural insights through an interactive Streamlit dashboard.
🎯 Project Objectives
Build a machine learning model to predict crop yield using weather, soil, crop, and seasonal information.
Compare Linear Regression, Random Forest, and XGBoost regression models.
Analyze how rainfall, temperature, humidity, and soil nutrients influence crop productivity.
Develop a crop water-stress alert system to identify potential irrigation needs.
Create district-level choropleth maps to visualize regional yield performance.
Develop an interactive dashboard for agricultural analytics and decision support.
Generate downloadable seasonal crop reports containing yield predictions, risk factors, and irrigation recommendations.
✨ Key Features
1. 🌱 Crop Yield Prediction

Users can select a crop, district, and season to estimate the expected crop yield.

Inputs:

Crop type
District or region
Growing season
Total growing-season rainfall
Average temperature
Growing degree days (GDD)
Soil nitrogen (N), phosphorus (P), and potassium (K)

Outputs:

Predicted yield in tons per hectare
Estimated prediction interval
Major factors influencing the prediction
Potential weather and soil-related risks
2. 💧 Smart Irrigation Advisory

The system evaluates weather conditions and crop water requirements to recommend irrigation actions.

Example advisory:

Today's forecast indicates a temperature of 35°C with no expected rainfall. Check soil moisture and crop water requirements before irrigating. The system may recommend a net irrigation depth of 15 mm if supported by the crop water balance.

The advisory considers:

Temperature and humidity
Forecast rainfall
Crop type and growth stage
Crop water requirements
Estimated crop water stress
Soil moisture or water-balance information, when available
3. 📊 Weather–Yield Correlation Analysis

Explore how weather conditions influence crop productivity through interactive visualizations.

Rainfall versus crop yield
Temperature versus crop yield
Humidity versus crop yield
Crop-wise and season-wise comparisons
Correlation heatmaps
Historical weather and yield trends
4. 🗺️ Regional Production Mapping

Visualize district-level crop yield and production patterns across India.

High-yield and low-yield regions
Crop-wise production distribution
District-level yield comparisons
Seasonal production patterns
Interactive choropleth maps
5. 🧪 Soil Nutrient Analysis

Study the relationship between soil nutrient levels and agricultural productivity.

Nitrogen (N) distribution
Phosphorus (P) distribution
Potassium (K) distribution
Soil nutrient comparisons across crops
Relationship between nutrient levels and predicted yield
6. 🔍 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) helps explain individual model predictions.

The dashboard will identify factors that increase or decrease a predicted yield, such as:

Lower-than-usual growing-season rainfall
High temperatures during critical growth stages
Favorable soil nutrient levels
Differences between crops, districts, and seasons
7. 📄 Seasonal Crop Report

Generate a downloadable PDF report containing:

Crop and district information
Predicted seasonal yield
Prediction interval and model information
Weather and soil risk factors
SHAP-based prediction explanations
Irrigation advisory summary
🛠️ Technology Stack
Category	Technologies
Programming Language	Python
Data Processing	Pandas, NumPy
Data Visualization	Plotly, Matplotlib, Seaborn
Machine Learning	scikit-learn, XGBoost
Explainable AI	SHAP
Web Application	Streamlit
Geospatial Visualization	Folium, streamlit-folium
Weather Data	Open-Meteo Historical Weather API
Data Sources	Kaggle, India Open Government Data Platform
Report Generation	Python PDF generation library
Version Control	Git, GitHub
📂 Project Structure
crop-yield-irrigation/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_weather_yield_analysis.ipynb
│   ├── 03_soil_nutrient_analysis.ipynb
│   ├── 04_yield_model_training.ipynb
│   └── 05_irrigation_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict_yield.py
│   ├── irrigation_advisor.py
│   ├── shap_explainer.py
│   └── report_generator.py
│
├── models/
│   ├── yield_model.pkl
│   └── preprocessor.pkl
│
├── dashboard/
│   ├── yield_dashboard.py
│   ├── irrigation_dashboard.py
│   ├── analytics_dashboard.py
│   └── regional_map.py
│
├── reports/
│   └── model_comparison.md
│
└── tests/
    ├── test_preprocessing.py
    └── test_predictions.py


Note: This is the proposed project structure. Model files, datasets, and modules will be added as development progresses. Large datasets and generated model artifacts should generally not be committed directly to GitHub.

📚 Dataset Sources
1. India Crop Production Dataset

Source: Kaggle Datasets

Used for:

Historical crop production records
Crop-wise and district-wise analysis
Seasonal and annual production trends
Historical yield calculations, where area data is available
2. Crop Recommendation Dataset

Source: Kaggle Datasets

Used for:

Soil nitrogen, phosphorus, and potassium values
Temperature and humidity observations
Rainfall and soil pH information
Exploratory soil and crop suitability analysis

Important: Crop recommendation datasets may not contain measured historical yields. Their soil and weather observations should not be treated as directly linked yield labels unless a valid matching relationship exists.

3. Weather Data

Source: Open-Meteo Historical Weather API

Used for:

Historical daily rainfall
Temperature observations
Relative humidity, where available
Growing-season weather summaries
Weather feature engineering
4. Additional Agricultural Resources
India Open Government Data Platform
FAO Crop Water Information
SHAP Documentation
Streamlit Documentation
⚙️ Methodology
Step 1: Data Collection

Collect historical crop production, cultivated area, weather, and soil nutrient data from the available sources.

Step 2: Data Cleaning and Integration
Handle missing values and duplicate records.
Standardize crop names, district names, and season labels.
Validate units and measurement ranges.
Aggregate daily weather data into crop-growing-season features.
Merge datasets using appropriate geographical, temporal, and crop identifiers.
Document missing data and assumptions.
Step 3: Exploratory Data Analysis

Analyze:

Crop yield trends over the past 20 years, where records permit.
Districts with the highest and lowest yields.
Weather–yield relationships.
Soil nutrient distributions.
Regional and seasonal production patterns.
Step 4: Feature Engineering

Potential features include:

Total rainfall during the growing season
Average, minimum, and maximum temperature
Relative humidity statistics
Growing degree days
Soil nitrogen, phosphorus, and potassium
Soil pH
Crop type, district, and season
Crop water requirements and growth stage, for irrigation
Step 5: Model Training

Train and compare the following regression algorithms:

Linear Regression — baseline model
Random Forest Regressor — nonlinear model
XGBoost Regressor — gradient-boosting model

Evaluate the models using:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Coefficient of Determination (R²)

Use chronological or grouped validation where appropriate to reduce information leakage between years, districts, and repeated observations.

Step 6: Model Explainability

Use SHAP to identify the main features contributing to predictions and visualize their effects at both the dataset and individual-prediction levels.

Step 7: Irrigation Advisory Development

Estimate crop water stress and irrigation requirements from weather, crop stage, and available water-balance inputs.

The irrigation logic should distinguish between:

Forecast rainfall
Crop evapotranspiration
Effective rainfall
Available soil moisture
Net irrigation requirement
Gross irrigation requirement, accounting for irrigation efficiency when known
Step 8: Dashboard Development

Build a Streamlit application with interactive pages for yield prediction, irrigation advice, weather–yield analysis, soil nutrient profiling, and regional production maps.

Step 9: Reporting and Deployment

Generate seasonal crop reports, document model limitations, and deploy the application using an appropriate hosting platform.

🚀 Installation and Setup
Prerequisites
Python 3.10 or a compatible version supported by the selected libraries
Git
A code editor such as Visual Studio Code
Internet access for retrieving weather data
1. Clone the Repository
git clone https://github.com/YOUR-USERNAME/crop-yield-irrigation.git
cd crop-yield-irrigation


Replace YOUR-USERNAME with your GitHub username.

2. Create a Virtual Environment
python -m venv .venv


Activate it:

Windows

.venv\Scripts\activate


macOS/Linux

source .venv/bin/activate

3. Install Dependencies

Create a requirements.txt file with the following initial dependencies:

streamlit
pandas
numpy
scikit-learn
xgboost
shap
plotly
matplotlib
seaborn
folium
streamlit-folium
requests
joblib
geopandas
reportlab


Install the packages:

pip install -r requirements.txt


Use compatible package versions and pin the tested versions before deployment.

4. Prepare the Datasets

Place downloaded or generated datasets in the appropriate folders:

data/raw/
data/processed/
data/external/


Check the dataset licenses and terms before redistribution. Keep API credentials and private configuration out of version control.

5. Run the Application

After creating the Streamlit entry point:

streamlit run app.py


The application will display a local URL in the terminal.

📈 Model Evaluation

The models will be compared using a common validation strategy.

Model	MAE	RMSE	R²
Linear Regression	To be measured	To be measured	To be measured
Random Forest	To be measured	To be measured	To be measured
XGBoost	To be measured	To be measured	To be measured

The final model will be selected based on validation performance, generalization across districts and years, and suitability for practical use.

Prediction intervals should be estimated using a validated method rather than assumed from a single point prediction.

👥 Team Responsibilities
Data Science & Data Analytics Team
Explore historical crop yield and production records.
Analyze weather–yield correlations.
Study soil nutrient distributions and their relationships with yield.
Develop district-level choropleth maps.
Build interactive agricultural analytics dashboards.
Prepare data visualizations and analytical reports.
AI/ML Team
Develop the data preprocessing pipeline and predictive features.
Train Linear Regression, Random Forest, and XGBoost models.
Evaluate and compare model performance.
Implement SHAP-based explanations.
Develop crop water-stress indicators.
Build daily irrigation advisory logic.
Integrate trained models with the Streamlit application.
Joint Responsibilities
Dataset integration and validation
Feature selection and evaluation strategy
Dashboard integration
Seasonal report generation
Testing, documentation, and final demonstration
🗓️ Development Roadmap
Week 1 — Data Preparation and Baseline Model
Collect crop production and weather datasets.
Clean and standardize the datasets.
Explore crop yield trends and regional patterns.
Engineer initial weather and soil features.
Train a Linear Regression baseline.
Week 2 — Advanced Analytics and Machine Learning
Develop regional production maps.
Analyze weather–yield correlations.
Study soil nutrient relationships.
Train Random Forest and XGBoost models.
Generate SHAP explanations.
Implement initial irrigation advisory rules.
Week 3 — Streamlit Application
Develop the crop yield prediction interface.
Integrate the trained model.
Build the daily irrigation advisory page.
Add interactive weather analytics.
Integrate district-level maps.
Freeze the minimum viable product (MVP).
Week 4 — Validation and Final Delivery
Compare model performance and document results.
Refine crop water-stress analysis.
Test data processing and predictions.
Implement downloadable seasonal reports.
Document limitations and assumptions.
Prepare the final project demonstration.
⚠️ Important Considerations
Data compatibility: District-level production records and crop recommendation datasets may not share matching locations, years, or individual fields. They must not be merged as though they represent the same observations without evidence.
Yield calculation: Yield should be calculated as production divided by harvested area, using consistent units.
Data leakage: Weather summaries and soil values must reflect information legitimately available for the prediction being made.
Prediction uncertainty: Report a confidence or prediction interval only when a suitable method has been implemented and validated.
Irrigation accuracy: Temperature and humidity alone cannot determine the exact water requirement of a field. Reliable advisories should also account for crop stage, evapotranspiration, rainfall, soil water availability, and irrigation efficiency when available.
Regional mapping: District boundary files must use consistent geographic identifiers and administrative boundaries.
Decision support: Predictions and irrigation recommendations are estimates, not guarantees. Field conditions and local agricultural guidance should be considered before acting.
🔮 Future Enhancements
Integrate real-time weather forecasts.
Incorporate satellite imagery and vegetation indices.
Add crop growth-stage detection.
Integrate soil moisture sensor data.
Support additional crops and geographical regions.
Add multilingual support for farmers.
Develop SMS or mobile notifications for irrigation alerts.
Integrate agricultural market prices and profitability estimates.
🤝 Contributing

Contributions are welcome!

Fork the repository.
Create a feature branch.
Commit your changes with clear messages.
Push the branch to your fork.
Open a pull request describing your changes.
📜 License

Choose an appropriate open-source license before publishing the repository. For example, the MIT License may be suitable if you want others to reuse and modify your code, subject to its terms.

🙌 Acknowledgments

We acknowledge the open-data and open-source communities supporting agricultural research, including Kaggle contributors, Open-Meteo, India's Open Government Data Platform, the FAO, scikit-learn, XGBoost, SHAP, Streamlit, Plotly, and Folium.

Built with Python, Data Science, and Machine Learning to support smarter agricultural decisions. 🌱
