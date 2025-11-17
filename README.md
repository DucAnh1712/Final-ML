# PROJECT REPORT: SAIGON (HCMC) TEMPERATURE FORECASTING

  * **Team:** Group 6
  * **Subject:** Subject 2: Saigon Temperature Forecasting

-----

## 1\. Project Objective (Step 1: Data Acquisition)

The core objective of this project is to build a complete Machine Learning system capable of forecasting the average daily temperature in Ho Chi Minh City (Saigon) for the **next 7 days** ($T+1$ to $T+7$).

To achieve this, we began by collecting 10 years of historical daily weather data (Jan 2015 - Dec 2025) from Visual Crossing, resulting in a raw dataset of 3,934 entries and 33 features.

-----

## 2\. Project Structure

The project is organized into a modular MLOps pipeline, separating concerns from data analysis to model training and deployment:

```
Final-ML/
├── daily/
│   ├── data/
│   │   ├── raw_data/
│   │   │   └── HCMWeatherDaily.xlsx    # Raw 10-year data (Input)
│   │   └── processed_data/
│   │       ├── data_train.csv          # 70% of data
│   │       ├── data_val.csv            # 15% of data
│   │       └── data_test.csv           # 15% of data
│   ├── src/
│   │   ├── visualize_weather.py        # Step 2: EDA script
│   │   ├── data_processing.py          # Step 3: Data splitting script
│   │   ├── feature_engineering.py      # Step 4: Feature pipeline
│   │   ├── benchmark.py                # Step 5: Model comparison
│   │   ├── optuna_search_linear.py     # Step 5: Hyperparameter tuning
│   │   ├── train.py                    # Step 5: Final model training
│   │   ├── inference.py                # Step 5: Final model evaluation
│   │   ├── visualize_results.py        # Step 5: Final model visualization
│   │   └── inference.py                # Step 5: Final model evaluation
│   ├── plots/                        # Output for visualize_weather.py
│   ├── models/                       # Output for optuna_search_linear.py
│   └── inference_results/            # Metrics for train.py and inference.py
├── hourly/                           # Code for Step 8
├── README.md                         # This report
├── requirements.txt                  # Project dependencies
└── scaler.pkl                        # Saved scaler from training
```

-----

## 3\. How to Run

### Requirements

Install the necessary libraries:

```bash
pip install -r requirements.txt
```

*(Note: `requirements.txt` should contain `pandas`, `matplotlib`, `seaborn`, `openpyxl`, `scikit-learn`, `optuna`, `clearml`, `lgbm`, `xgboost`, `pyyaml`)*

### Execution Steps

**1. Run Exploratory Data Analysis (EDA):**
This script reads the raw Excel file and generates all analysis plots in the `/daily/plots/` folder.

```bash
python daily/src/visualize_weather.py
```

**2. Run Data Processing (ML Preparation):**
This script reads the raw Excel, creates the 7-day targets, and splits the data by time into `train/val/test` CSVs.

```bash
python daily/src/data_processing.py
```

-----

## 4\. Step 2: Exploratory Data Analysis (EDA)

Before building any models, we ran `visualize_weather.py` to deeply understand the data, as detailed in the project notebook. This step answers the core questions from the project brief.

### 4.1. Data Dictionary (Understanding the Features)

The brief asks: *"Explain the meaning and values in each column. For example, what does feature 'moonphase' mean?"*.

The dataset has 33 features. While most are self-explanatory (e.g., `temp`, `humidity`, `precip`), key specialized columns include:

  * **`dew` (Dew Point):** The temperature at which air becomes 100% saturated and dew forms.
  * **`solarradiation` (Solar Radiation):** Average solar energy received, measured in $W/m^{2}$.
  * **`moonphase` (Moon Phase):** A value from 0 to 1 representing the lunar cycle:
      * `0.0`: New Moon
      * `0.5`: Full Moon
      * `0.75`: Last Quarter

### 4.2. Target Column Analysis (The 10-Year Trend)

The brief asks: *"Try to plot the target column... What is your observation about... the last 10 years?"*.

[cite\_start]We plotted the target column, `temp` (Average Daily Temperature), over the 10-year period [cite: 79, 81-88].

*Plot `daily_temp_timeseries.png`: The target variable (`temp`) over 10 years.*

**Observations:**

1.  **Clear Seasonality:** A strong, consistent annual cycle is visible. The temperature peaks every year around April/May and hits its low point in December/January.
2.  **Stable Range:** The temperature oscillates in a very tight and predictable range, almost entirely between 24°C and 33°C.

We also plotted the average annual temperature to check for a long-term trend.

*Plot `annual_avg_temp.png`: Checking for long-term trends.*

This plot shows significant year-to-year variation, with a cool year in 2017 (28.1°C) and a sharp peak in 2024 (29.1°C). Overall, the data suggests a **slight warming trend** across the decade.

### 4.3. Annual & Seasonal Trends

[cite\_start]The data clearly shows two distinct seasons: a **Dry Season** (Dec-Apr) and a **Rainy Season** (May-Nov) [cite: 238, 241, 800-802]. We used boxplots to visualize the difference:

*Plot `seasonal_boxplots.png`: This perfectly summarizes HCMC's climate.*

  * **Dry Season (Left):** Characterized by lower humidity, almost zero rainfall, and slightly higher average temperatures.
  * **Rainy Season (Right):** Characterized by extremely high humidity, frequent heavy rainfall, and slightly cooler, more stable temperatures.

This seasonal pattern is the most dominant feature of the dataset.

### 4.4. Correlation Analysis (Answering "How they combine")

The brief asks: *"Try to understand the relationship between different features... (their correlation maybe). How they can combine together to detect... temperature."*.

We generated a correlation matrix (heatmap) and scatter plots to find the key drivers of temperature.

*Plot `correlation_heatmap.png`: The relationship between all 33 features.*

**Key Findings:**

1.  **High Multicollinearity:** There are two groups of redundant features:

      * **Temperature Group:** `temp`, `tempmax`, `tempmin`, and `feelslike` are all correlated at $r > 0.90$.
      * **Solar Group:** `solarradiation`, `solarenergy`, and `uvindex` are correlated at $r \approx 1.00$.
      * **Action:** We must *not* use all these features in our model. [cite\_start]We will select one representative from each group [cite: 377-380].

2.  **Key Predictors (How they combine):**

      * **`humidity` (Negative):** The strongest driver. As humidity goes up, temperature goes down. The scatter plot shows a clear negative trend ($r \approx -0.82$).
      * **`solarradiation` (Positive):** As solar radiation increases, temperature increases.
      * **`precip` (Negative):** Hot days (\>30°C) almost never have rain, while high-rainfall days are almost always cooler.

**Conclusion:** The EDA tells us that temperature is driven by a combination of **strong seasonality** (time of year) and a physical battle between **solar radiation** (heating) and **humidity/rain** (cooling).

-----

## 5\. Step 3 & 4: The Data Processing & Engineering Pipeline

Based on our EDA findings, we built a robust pipeline to process data and engineer features.

### 5.1. Data Processing (Step 3)

This step, executed by `data_processing.py`, prepares the data for modeling.

  * **Target Creation:** As our goal is a 7-day forecast, we create 7 separate target columns (`target_t1`...`target_t7`) by shifting the `temp` column backwards = df\_new[target\_col].shift(-h)].
  * **Time-Aware Data Splitting:** This is the most important step to prevent data leakage. We **do not shuffle**. Instead, we split the data strictly by time:
      * **Train:** The first 70% of the data.
      * **Validation:** The next 15% of the data.
      * **Test:** The final 15% of the data.
  * **Handle Missing Values:** The script handles missing data (like `preciptype` and `severerisk`) by filling them, and the final `ColumnPreprocessor` handles any remaining NaNs using `ffill`/`bfill`.

### 5.2. Feature Engineering (Step 4)

Using the `feature_engineering.py` script, we created a `sklearn.pipeline` to automatically transform the raw data.

1.  **`TimeFeatureTransformer`:** Addresses **seasonality**. It converts the `datetime` index into cyclical `sin`/`cos` features for the month and day-of-year = np.sin... This teaches the model that December (12) is "close" to January (1).
2.  **`DerivedFeatureTransformer`:** Creates new physics-based features from existing ones, such as `temp_range` (`tempmax` - `tempmin`) and `dewpoint_depression` (`temp` - `dew`).
3.  **`ColumnPreprocessor`:** This is the **anti-leakage** and **anti-multicollinearity** step. It selects only the useful features identified in our EDA and, most importantly, **removes the original `temp` column**, preventing the model from cheating by looking at the answer.

-----

## 6\. Step 5: Model Training & Tuning (Daily Data)

This step answers the question: "Which model is best for the Daily data?"

### 6.1. The Daily Benchmark (`benchmark.py`)

We first held a "competition" with 6 common models to see which performed best on our V4 features.

**The results were shocking:**

| Model | Train RMSE | Test RMSE | Gap (Overfit) |
| :--- | :--- | :--- | :--- |
| **LinearRegression** | 0.774 | **0.820** | **5.9%** |
| **Ridge** | 0.775 | **0.823** | **6.2%** |
| RandomForest | 0.568 | 0.939 | 65.4% |
| LightGBM | 0.468 | 0.902 | 92.7% |
| XGBoost | 0.459 | 0.898 | 95.8% |
| DecisionTree | 0.000 | 1.242 | \~Very Large |

**Conclusion:** The complex, tree-based models (XGBoost, LGBM) **failed completely**. They overfit the training data (e.g., 95.8% Gap for XGBoost). The simple **Linear Models (LinearRegression, Ridge) won** by a huge margin. This proves our V4 (Daily) features have a strong *linear* relationship with temperature.

### 6.2. Hyperparameter Tuning (`optuna_search_linear.py`)

Having crowned Linear Models as our champion, we used Optuna to find the best possible version. To do this safely, we used a custom `PurgedTimeSeriesSplit`. This custom cross-validation:

1.  Splits the data by time.
2.  Leaves a 7-day **"Gap"** (or "purge") between the train and val sets.

This 7-day gap is critical: it matches our 7-day forecast horizon, ensuring that no information from the validation period can *ever* leak into the training period, making our parameter search 100% reliable.

### 6.3. Final Daily Model & Metrics Interpretation

We ran `train_linear.py` to train 7 final models (one for each day, T+1 to T+7) on 85% of the data. The brief asks us to "use them all [metrics], understand and interprete them".

**Here is the T+1 (1-Day Forecast) Performance on Unseen Test Data:**

  * **RMSE: 0.828°C**

      * **Interpretation:** This is our primary metric. On average, our model's prediction is wrong by only **0.828 degrees Celsius**. This is a highly accurate result.

  * **R² (R-Squared): 0.724** (or 72.4%)

      * **Interpretation:** Our model is able to **explain 72.4%** of the variance (the "change") in the daily temperature.

  * **MAPE (Mean Absolute Percentage Error): 2.31%**

      * **Interpretation:** This is the easiest to explain. It means our forecast is, on average, only **2.31%** off from the actual temperature.

-----

## 7\. Step 6: Application UI (Streamlit)

To make our model accessible, we built a web application using Streamlit. This app loads the saved `feature_pipeline.pkl`, `scaler.pkl`, and the 7 trained model files (`.pkl`) to deliver a 7-day temperature forecast to the end-user in an interactive interface.

-----

## 8\. Step 7: Retraining Strategy (When to Retrain?)

The brief asks: *"if you... predict day by day... performance will downgrade. When you should retrain your model?"*.

First, we must understand *why* retraining is necessary. A model may perform well today, but its performance can degrade over time. This is known as **Model Drift**.

Model Drift occurs when the statistical properties of the input data change, violating the original assumptions the model was trained on. This is not a change in the model, but a change in the *environment*.

For our HCMC temperature model, drift is **guaranteed** due to:

  * **Seasonality:** The clear shift between the rainy season (May-Nov) and dry season (Dec-Apr).
  * **Anomalies:** Unpredictable weather events like El Niño/La Niña.
  * **Long-term Trends:** Climate change ensures that data from 2025 will be different from data from 2015.

Our team has decided on a **hybrid strategy** to combat model drift, combining threshold-based triggers and periodic retraining.

### 8.1. Method 1: Metric-Based Trigger (Threshold)

This method involves retraining only when performance drops below a set threshold.

  * **Baseline:** We will use the metrics from our `test_metrics_linear.yaml` file as the baseline. Our baseline for the T+1 forecast is an **RMSE of 0.828°C**.
  * **Monitoring:** In production, we will calculate the 30-day rolling average RMSE for *each* of our 7 targets (T+1 to T+7).
  * **Trigger:** If the 30-day rolling RMSE for **any single target** deviates from its baseline by a set threshold (e.g., 20%), a full retraining of all 7 models is automatically triggered. A failure in one model suggests the entire feature pipeline is drifting.

### 8.2. Method 2: Periodic Trigger (Scheduled)

This method involves retraining at a fixed interval, regardless of performance.

  * **Rationale:** Weather data is constantly changing due to global warming and climate change. We must proactively retrain to ensure the model does not become obsolete.
  * **Schedule:** We propose retraining the model every **6 months** (approx. 180 new daily records).
  * **Timing:** This schedule is designed to capture seasonal transitions (e.g., around April/May and Nov/Dec). This allows the model to learn the most recent patterns of the just-ended dry or rainy season, which may differ significantly from previous years.

### 8.3. Final Hybrid Strategy

We will combine both methods for a robust system:
**We will retrain the model every 6 months, regardless of performance. If a metric-based trigger (Method 1) fires *before* the 6-month mark, we will retrain immediately.**

-----

## 9\. Step 8: The Hourly Data Experiment

The brief asks: *"with hourly data... can you do better?"*. We re-ran the *entire* MLOps pipeline using the **Hourly Data**. This dataset is much larger (24x) and contains more high-frequency noise.

### 9.1. The Hourly Benchmark (A Different Story)

Unlike the Daily data (where Linear models won), the initial benchmark on Hourly data told a completely different story. We ran the same 6 model types:

| Model | Test RMSE | Overfitting (Gap %) | Analysis |
| :--- | :--- | :--- | :--- |
| **LightGBM** | **1.498** | 18.3% | **Best Accuracy** |
| **XGBoost** | **1.499** | 17.4% | **2nd Best Accuracy** |
| **LinearRegression** | 1.511 | **3.2%** | **Best Stability** |
| **Ridge** | 1.511 | **3.2%** | (Same as Linear) |
| RandomForest | 1.520 | 19.3% | (Worse than LGBM/XGB) |
| DecisionTree | 2.134 | \~Huge | (Overfit) |

**Conclusion:** The benchmark clearly showed that for Hourly data, **Tree-Based Models (LGBM, XGB) are the most accurate**, while **Linear Models are the most stable** (least overfit). This justified fine-tuning all three families (Linear, XGBoost, LightGBM) for a final comparison.

### 9.2. Final Tuned Model Comparison (The Showdown)

After running all three models through our `optuna_search` pipeline, we compared their final, tuned performance on the Test set.

#### A. Absolute Accuracy (Test RMSE)

This chart compares the average prediction error (RMSE) for the 7-day forecast. **Lower is better.**

*Plot `image_e3bdc2.png`: Test RMSE comparison of the three tuned models.*

**Findings:**

1.  **Linear Model:** The (blue) line shows the Linear model is **the least accurate** at every single forecast horizon (T+1 RMSE \~1.51°C).
2.  **Tree Models:** Both **LightGBM (green)** and **XGBoost (orange)** are significantly more accurate, starting with an RMSE of \~1.48°C.
3.  **LGBM vs. XGB:** A close look shows **LightGBM is the winner for short-term forecasts** (Days 1-6), while **XGBoost pulls ahead slightly for the long-term** (Day 7).

#### B. Model Fit (Test R²)

This chart shows how much of the temperature's variance the model can explain. **Higher is better.**

*Plot `image_e3bb1c.png`: Test R² (R-Squared) comparison.*

**Findings:**

  * This plot confirms the RMSE results. The **Linear** model's R² (blue line) drops significantly over time, falling to \~67% by Day 7.
  * Both **XGBoost and LightGBM** are far more stable, maintaining a high R² of **over 70%** even at 7 days out. This proves they are better at modeling the complex, non-linear patterns in the hourly data.

### 9.3. Final Conclusion (Step 8)

1.  **Hourly Winner:** For hourly data, **Tree-Based Models (LGBM/XGB) are superior** to Linear models in accuracy and stability (R²). The best choice depends on the horizon: **LightGBM for short-term (T+1 to T+6)** and **XGBoost for long-term (T+7)**.
2.  **Overall Winner (Daily vs. Hourly):**
      * **For pure *Accuracy* (lowest RMSE):** The **Daily Linear Model** is still the champion (Test RMSE T+1 of **0.828** beats the Hourly LGBM's \~1.48).
      * **For long-term *Stability* (best R²):** The **Hourly Tree Models** are the winners. Their T+7 R² of **\~70%** is far more reliable than the Daily model's T+7 R² of **43%**.

-----

## 10\. Step 9: Deployment with ONNX

The final step, currently in progress, is to study ONNX (Open Neural Network Exchange). The goal is to convert our 7 trained `scikit-learn` models (saved as `.pkl` files) into the `.onnx` format. This will optimize them for high-speed inference and make them platform-independent, allowing them to be deployed on any server, edge device, or web service, regardless of the original Python environment.