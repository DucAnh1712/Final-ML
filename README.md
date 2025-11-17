# PROJECT REPORT: SAIGON (HCMC) TEMPERATURE FORECASTING

  * **Team:** Group 6
    **Member:**
     * Nguyễn Thị Mai Anh - 11230511
     * Trương Đức Anh - 11230516
     * Phạm Khánh Dương -	11230525
     * Nguyễn Thị Hương Giang -	11230532
     * Nguyễn Văn Khôi -	11230550

  * **Subject:** Saigon Temperature Forecasting

* **Project Framework:** This MLOps pipeline is developed entirely in Python, utilizing key libraries such as **Scikit-learn** for modeling, **Optuna** for hyperparameter tuning, and **Streamlit** for the deployment interface.

-----

## 1\. Project Objective (Step 1: Data Acquisition)

The core objective of this project is to build a complete Machine Learning system capable of forecasting the average daily temperature in Ho Chi Minh City (Saigon) for the **next 7 days** ($T+1$ to $T+7$).

To achieve this, we began by collecting **over 10 years** of historical daily weather data, specifically from **January 1, 2015, to October 8, 2025**, from Visual Crossing. This resulted in a raw dataset of **3,934 entries** and **33 features**.

-----

## 2\. Project Structure

The project is organized into a modular MLOps pipeline, separating concerns from data analysis to model training and deployment:

```
Final-ML/
├── daily/
│   ├── raw_data/
│   │   └── HCMWeatherDaily.xlsx    # Raw 10-year data (Input)
│   └── processed_data/
│       ├── data_train.csv          # 70% of data
│       ├── data_val.csv            # 15% of data
│       └── data_test.csv           # 15% of data
│   ├── src/
│   │   ├── visualize_weather.py        # Step 2: EDA script
│   │   ├── data_processing.py          # Step 3: Data splitting script
│   │   ├── feature_engineering.py      # Step 4: Feature pipeline
│   │   ├── benchmark.py                # Step 5: Model comparison (Linear, XGB...)
│   │   ├── optuna_search_linear.py     # Step 5: Hyperparameter tuning
│   │   ├── train.py                    # Step 5: Final model training
│   │   ├── inference.py                # Step 5: Final model evaluation
│   │   ├── visualize_results.py        # Step 5: Final model visualization
│   │   ├── assets/
│   │   |   ├── day_sky_background.png      # Background of daily Streamlit
│   │   ├── app.py                      # Streamlit UI code
│   │   ├── convert_to_onnx.py          # Step 9: Script convert to ONNX
│   │   └── benchmark_onnx.py           # Step 9: ONNX speed comparison
│   ├── plots/                      # Output for visualize_weather.py
│   │   ├── correlation_heatmap.png
│   │   ├── daily_temp_timeseries.png
│   │   └── ...
│   ├── models/                     # Output for optuna_search_linear.py
│   │   ├── feature_pipeline.pkl
│   │   ├── onnx_convertible_pipeline.pkl
│   │   ├── optuna_best_params_linear.yaml
│   │   ├── scaler.pkl              # Saved scaler from training
│   │   ├── target_t1_model_linear.pkl
│   │   ├── target_t1_model_linear.pkl.onnx
│   │   └── ... (7 models .pkl and .onnx)
│   └── inference_results/          # Metrics for train.py and inference.py
│   │   ├── benchmark_results.yaml
│   │   ├── inference_benchmark.yaml
│   │   ├── rmse_by_horizon.png
│   │   ├── test_metrics_linear.yaml
│   │   └── ...
├── hourly/                     # Code for Step 8
├── project_subjects.pdf       
├── README.md                   # This report
└── requirements.txt            # Project dependencies
```

-----
## 3\. How to Run

This guide provides the complete step-by-step instructions to replicate the project, from data processing to final application.

### 3.1. Environment Setup

First, create a virtual environment and install all required dependencies.

1.  **Create a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    # For Windows
    python -m venv .venv
    ```
2.  **Activate the environment:**
    ```bash
    # For macOS/Linux
    source .venv/bin/activate
    # For Windows
    .\.venv\Scripts\activate
    ```
3.  **Install the necessary libraries:**
    ```bash
    pip install -r requirements.txt
    ```


### 3.2. Main Execution Pipeline

Follow these steps in order. Each script processes data and saves its output, which is then used by the next script.

1.  **Run Exploratory Data Analysis (EDA):**
    This script reads the raw Excel file and generates all analysis plots (e.g., `correlation_heatmap.png`, `daily_temp_timeseries.png`) in the **`/daily/plots/`** folder.

    ```bash
    python daily/src/visualize_weather.py
    ```

2.  **Run Data Processing:**
    This script reads the raw Excel, creates the 7-day targets, and splits the data by time into `data_train.csv`, `data_val.csv`, and `data_test.csv` inside the **`/daily/processed_data/`** folder.

    ```bash
    python daily/src/data_processing.py
    ```

3.  **Run Model Benchmarking:**
    This script trains 6 different models (Linear Regression, XGBoost, etc.) on the T+1 data to find the best model type. Results are saved in **`/daily/inference_results/benchmark_results.yaml`**.

    ```bash
    python daily/src/benchmark.py
    ```

4.  **Run Hyperparameter Tuning:**
    Based on the benchmark, this script uses Optuna to fine-tune the chosen model (Linear Regression) for all 7 forecast horizons. The best parameters are saved in **`/daily/models/optuna_best_params_linear.yaml`**.

    ```bash
    python daily/src/optuna_search_linear.py
    ```

5.  **Run Final Model Training:**
    This script loads the best parameters from Optuna and trains the 7 final models (one for each day, T+1 to T+7). The final models (`.pkl`), scaler, and pipelines are saved in the **`/daily/models/`** folder.

    ```bash
    python daily/src/train.py
    ```

6.  **Run Final Inference & Evaluation:**
    This script loads the 7 trained models and evaluates them on the `data_test.csv` set. Final metrics (e.g., `test_metrics_linear.yaml`) and predictions are saved in **`/daily/inference_results/`**.

    ```bash
    python daily/src/inference.py
    ```


### 3.3. Running the Application & Optimization

1.  **Run the Streamlit Web App:**
    This command starts the local web server and opens the Streamlit application in your browser to interact with the trained models.

    ```bash
    streamlit run daily/src/app.py
    ```

2.  **Run ONNX Conversion & Benchmark (Optional):**
    These steps are for optimizing and testing deployment speed.

    ```bash
    # Step 1: Convert the 7 trained .pkl models to .onnx format
    python daily/src/convert_to_onnx.py

    # Step 2: Run a speed test comparing Sklearn vs. ONNX inference time
    python daily/src/benchmark_onnx.py
    ```

-----
## 4. Step 2: Exploratory Data Analysis (EDA)

Before building any models, we ran `visualize_weather.py` to deeply understand the data.

### 4.1. Data Dictionary (Understanding the Features)

The dataset has 33 features. While most are self-explanatory (e.g., `temp`, `humidity`, `precip`), key specialized columns include:

* **`dew` (Dew Point):** The temperature at which air becomes 100% saturated and dew forms.
* **`solarradiation` (Solar Radiation):** Average solar energy received, measured in $W/m^2$.
* **`moonphase` (Moon Phase):** A value from 0 to 1 representing the lunar cycle:
    * `0.0`: New Moon
    * `0.5`: Full Moon
    * `0.75`: Last Quarter

### 4.2. Target Column Analysis (The 10-Year Trend)

We plotted the target column, `temp` (Average Daily Temperature), over the 10-year period.

![alt text](daily/plots/daily_temp_timeseries.png)

**Observations:**

1.  **Clear Seasonality:** A strong, consistent annual cycle is visible. The temperature peaks every year during the hot season (around **April/May**) and hits its low point in the cool season (**December/January**).
2.  **Stable Range:** The temperature oscillates in a very tight and predictable range, almost entirely between **24°C and 33°C**.

We also plotted the average annual temperature to check for a long-term trend.

![alt text](daily/plots/annual_avg_temp.png)

This plot shows significant year-to-year variation, with a cool year in 2017 (28.1°C) and a sharp peak in 2024 (29.1°C). Overall, the data suggests a **slight warming trend** across the decade.

#### 4.2.1. Distribution Analysis

![alt text](daily/plots/temp_distribution.png)

**Findings:**
1.  **Slightly Bimodal (Two slight peaks)**: There are two small peaks, one around 27.5°C (Rainy Season) and one around 30°C (Dry Season).
2.  **Left-Skewed:** The data is slightly skewed to the left, indicating there are slightly more cool days than extremely hot days.

### 4.3. Annual & Seasonal Trends

The data clearly shows two distinct seasons: a **Dry Season** (Dec-Apr) and a **Rainy Season** (May-Nov). We used boxplots to visualize the difference:

![alt text](daily/plots/seasonal_boxplots.png)

* **Dry Season (Left):** Characterized by lower humidity, almost zero rainfall, and slightly higher average temperatures.
* **Rainy Season (Right):** Characterized by extremely high humidity, frequent heavy rainfall, and slightly cooler, more stable temperatures.

This seasonal pattern is the most dominant feature of the dataset.

#### 4.3.1. Climograph Analysis

To confirm this cyclical relationship, we plotted a climograph.

![alt text](daily/plots/climograph_temp_humidity.png)

**Finding:** The chart shows a distinct **"looping relationship"**. It does not follow a straight line; instead, the weather "moves" through the seasons: from Hot & Dry (Q1) -> Hot & Humid (Q2) -> Cool & Humid (Q3/Q4) -> and back again. This is strong visual evidence that the weather is a cycle.

### 4.4. Correlation Analysis

We generated a correlation matrix (heatmap) and scatter plots to find the key drivers of temperature.

![alt text](daily/plots/correlation_heatmap.png)

**Key Findings:**

1.  **High Multicollinearity:** There are groups of redundant features:
    * **Temperature Group:** `temp`, `tempmax`, `tempmin`, and `feelslike` are all correlated at $r > 0.90$.
    * **Solar Group:** `solarradiation`, `solarenergy`, and `uvindex` are correlated at $r \approx 1.00$.
    * **Action:** We must *not* use all these features. We will select one representative from each group.

2.  **Key Predictors (How they combine):**
    * **`humidity` (Negative):** The strongest driver. As humidity goes up (mùa mưa), temperature goes down. The scatter plot (`scatter_temp_vs_humidity.png`) shows a clear negative trend ($r \approx -0.82$).
    * **`solarradiation` (Positive):** As solar radiation increases (mùa khô, trời nắng), temperature increases. Plot `scatter_temp_vs_solar.png` ($r \approx +0.70$).
    * **`precip` (Negative):** Hot days (>30°C) almost never have rain, while high-rainfall days are almost always cooler.

**Conclusion:** The EDA tells us that HCMC's temperature is driven by a combination of **strong (2-season) seasonality** (time of year) and a physical battle between **solar radiation** (heating) and **humidity/rain** (cooling).

-----

## 5\. Step 3 & 4: The Data Processing & Engineering Pipeline

Based on our EDA findings, we built a robust pipeline to process data and engineer features.

### 5.1. Data Processing (Step 3)

This step, executed by `data_processing.py`, prepares the data for modeling.

  * **Target Creation:** As our goal is a 7-day forecast, we create 7 separate target columns (`target_t1`...`target_t7`) by shifting the `temp` column backwards = df\_new[target\_col].shift(-h).
  * **Time-Aware Data Splitting:** This is the most important step to prevent data leakage. We **do not shuffle**. Instead, we split the data strictly by time:
      * **Train:** The first 70% of the data.
      * **Validation:** The next 15% of the data.
      * **Test:** The final 15% of the data.
  * **Handle Missing Values:** The script handles missing data (like `preciptype` and `severerisk`) by filling them, and the final `ColumnPreprocessor` handles any remaining NaNs using `ffill`/`bfill`.

Chắc chắn rồi! Dưới đây là phần **5.2. Feature Engineering (Step 4)** đã được sửa đổi và làm rõ chi tiết, được định dạng hoàn toàn bằng Markdown để bạn đưa vào báo cáo:

---

### 5.2. Feature Engineering (Step 4)

Using the `feature_engineering.py` script, we created a robust `sklearn.pipeline` to automatically transform the raw data based on physical insights from the EDA.

1.  **`TimeFeatureTransformer`:** Addresses **seasonality** and the cyclic nature of weather. It converts the `datetime` index into cyclical `sin`/`cos` features for the month and day-of-year. This teaches the model that December (12) is "close" to January (1) without abrupt transitions.
2.  **`DerivedFeatureTransformer`:** Creates new physics-based features from existing ones. This stage generates powerful predictors, including:
    * **`temp_range`:** Calculated as (`tempmax` - `tempmin`).
    * **`dewpoint_depression`:** Calculated as (`temp` - `dew`), a strong indicator tied directly to relative humidity.
    * **Solar Intensity:** Calculates **`daylight_hours`** and derives **`solar_per_hour`** (`solarenergy / daylight_hours`) to provide a normalized measure of daily solar intensity, which is superior to the total solar energy.
    * **`sealevelpressure_change`:** Calculates the pressure change between consecutive days.
3.  **`ColumnPreprocessor`:** This is the crucial **anti-leakage** and **anti-multicollinearity** step.
    * **Anti-Leakage:** It **removes the original `temp` column** from the feature set, preventing the model from cheating by looking at the answer.
    * **Anti-Multicollinearity:** Based on EDA's correlation analysis, it **selects only one representative** from highly correlated groups. For example:
        * **Temperature Group:** Redundant columns like `tempmax`, `tempmin`, and `feelslike` were excluded, relying instead on the derived feature **`temp_range`**.
        * **Solar Group:** We kept **`solarradiation`** as a primary measure but removed highly correlated columns like `solarenergy` and `uvindex` from the final feature set.
    * **Imputation:** It handles any remaining NaNs using a time-series safe imputation strategy (**Forward-fill, then Backward-fill**).

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

**Conclusion:** The complex, tree-based models (XGBoost, LGBM) **failed completely**. They overfit the training data (e.g., 95.8% Gap for XGBoost). The simple **Linear Models (LinearRegression, Ridge) won** by a huge margin. This proves our Daily features have a strong *linear* relationship with temperature.

### 6.2. Hyperparameter Tuning (`optuna_search_linear.py`)

Having crowned Linear Models as our champion, we used Optuna to find the best possible version. To do this safely, we used a custom `PurgedTimeSeriesSplit`. This custom cross-validation:

1.  Splits the data by time.
2.  Leaves a 7-day **"Gap"** (or "purge") between the train and val sets.

This 7-day gap is critical: it matches our 7-day forecast horizon, ensuring that no information from the validation period can *ever* leak into the training period, making our parameter search 100% reliable.

### 6.3. Final Daily Model & Metrics Interpretation

We ran `train_linear.py` to train 7 final models (one for each day, T+1 to T+7) on 85% of the data.

We will first analyze the **T+1 model** as our "best-case" scenario, and then analyze the **general trend** across all 7 models.

#### 6.3.1. Best-Case Performance: The T+1 (1-Day Forecast)

This model is our most accurate, as it predicts the nearest day.

* **RMSE: 0.828°C**
    * **Interpretation:** This is our primary metric. On average, our model's prediction for tomorrow's temperature is wrong by only **0.828 degrees Celsius**. This is a highly accurate result.

| Horizon | RMSE | MAE | R² | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **T+1** | 0.8281 | 0.6638 | 0.7243 | 2.31% |
| **T+2** | 1.0354 | 0.8453 | 0.5692 | 2.94% |
| **T+3** | 1.1223 | 0.9157 | 0.4942 | 3.18% |
| **T+4** | 1.1577 | 0.9413 | 0.4621 | 3.27% |
| **T+5** | 1.1688 | 0.9609 | 0.4522 | 3.33% |
| **T+6** | 1.1863 | 0.9654 | 0.4362 | 3.34% |
| **T+7** | 1.1938 | 0.9683 | 0.4299 | 3.35% |

* **R² (R-Squared): 0.724** (or 72.4%)
    * **Interpretation:** Our model is able to **explain 72.4%** of the variance (the "change") in the daily temperature.

* **MAPE (Mean Absolute Percentage Error): 2.31%**
    * **Interpretation:** This is the easiest to explain. It means our forecast is, on average, only **2.31%** off from the actual temperature.

#### 6.3.2. General Performance: The 7-Day Horizon Trend

When analyzing the models "in general", we observe a clear, logical, and expected trend as we forecast further into the future.

##### **Executive Summary of Model Performance**

The performance of the temperature forecasting model **naturally degrades** as the **forecast horizon (T+N)** increases, evidenced by the following key findings:

* **Error Increase (MAE & RMSE):** Both the Mean Absolute Error (**MAE**) and Root Mean Square Error (**RMSE**) increase steadily over time.
    * **T+1:** Lowest error (**MAE** $\approx 0.66^{\circ}C$, **RMSE** $\approx 0.83^{\circ}C$).
    * **T+7:** Highest error (**MAE** $\approx 0.96^{\circ}C$, **RMSE** $\approx 1.20^{\circ}C$).
* **Fit Decrease ($R^2$):** The model's explanatory power (Test $R^2$) drops significantly from **$\approx 72\%$** at T+1 to **$\approx 43\%$** at T+7.
* **Overfitting Gap:** A noticeable gap exists between the performance on the training set (Train) and the test set (Test) across all horizons, indicating the model shows signs of **overfitting**.

##### **Detailed Analysis of Charts**

1. Model Performance (MAE & RMSE)
![alt text](daily/inference_results/mae_by_horizon.png)
![alt text](daily/inference_results/rmse_by_horizon.png)

| Metric | T+1 (Best) | T+7 (Worst) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Test MAE** | $\approx 0.66^{\circ}C$ | $\approx 0.96^{\circ}C$ | The average forecast error increases by **$\approx 45\%$** when forecasting 6 days further out. |
| **Test RMSE** | $\approx 0.83^{\circ}C$ | $\approx 1.20^{\circ}C$ | Error increases, indicating a rise in larger errors (as RMSE penalizes large errors more severely than MAE). |

**Observation:** The error curve (orange line) is steepest from **T+1 to T+3**. After this point, the error continues to rise but at a slower rate (the curve begins to flatten).

2. Model Fit ($R^2$)
![alt text](daily/inference_results/r2_by_horizon.png)

| Forecast Day | Train $R^2$ | Test $R^2$ |
| :--- | :--- | :--- |
| **T+1** | $\approx 0.67$ | $\approx 0.72$ |
| **T+4** | $\approx 0.40$ | $\approx 0.46$ |
| **T+7** | $\approx 0.33$ | $\approx 0.43$ |

**Observation:**
* The Test $R^2$ is **higher** than the Train $R^2$ at T+1, which is **unusual** in regression analysis and may be due to differences in the distribution or complexity between the 85% training data and the 15% test data.
* The Test $R^2$ **drops sharply** from T+1 to T+4 (from $72\%$ to $46\%$), confirming that the model's predictive capability significantly diminishes after just 4 days.

3. Model Overfitting (Train-Test RMSE Gap)
![alt text](daily/inference_results/overfitting_gap_by_horizon.png)

This chart assesses the difference (as a percentage) between the RMSE on the Test set and the RMSE on the Train set:

| Forecast Day | Overfitting Gap (%) | Interpretation |
| :--- | :--- | :--- |
| **T+1** | $\approx 7.5\%$ | Lowest level of overfitting. |
| **T+4** | **$\approx 11.8\%$** | **Highest** level of overfitting observed. |
| **T+7** | $\approx 8.2\%$ | The level of overfitting slightly decreases. |

**Observation:**
* The Overfitting Gap **peaks** on **T+4**, suggesting the model is most over-fitted to the training data for this medium-range forecast horizon.
* The existence of a positive **Overfitting Gap** across all horizons confirms the model cannot perfectly generalize its performance from the training data to unseen (Test) data.

##### **Overall Conclusion**

The model performs **best** for **short-term forecasts (T+1 to T+2)** with high accuracy (MAE $\leq 0.75^{\circ}C$, $R^2 \geq 57\%$).

However, it exhibits a **rapid decline** in performance and a **significant degree of overfitting** as the forecast extends further out. Specifically, after T+4, although the error continues to rise, the reliability ($R^2$) is only about $45\%$, indicating this is the **practical limit** of this model's useful range.

-----

## 7\. Step 6: Application UI (Streamlit)

To make our model accessible, we built a web application using Streamlit. This app loads the saved `feature_pipeline.pkl`, `scaler.pkl`, and the 7 trained model files (`.pkl`) to deliver a 7-day temperature forecast to the end-user in an interactive interface.

You can access the live application here: https://aduongne-ml-final.hf.space

-----

## 8\. Step 7: Retraining Strategy

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

We re-ran the *entire* MLOps pipeline using the **Hourly Data**. This dataset is much larger (24x) and contains more high-frequency noise.

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

#### Executive Summary   
The analysis yields a consistent conclusion across all key metrics:

* The **Linear Model** is the **least effective** across all forecast horizons.
* **LightGBM** and **XGBoost** (tree-based models) show **superior and highly competitive performance**.
* **XGBoost** emerges as the overall **best performer** for long-term stability and accuracy (T+7).

1. Absolute Accuracy Comparison (Test MAE and Test RMSE)

![alt text](hourly/inference_results/compare_ALL_MODELS_Test_RMSE.png)
![alt text](hourly/inference_results/compare_ALL_MODELS_Test_MAE.png)

These metrics measure the average temperature prediction error. **Lower values are better.**

| Metric | T+1 (Short-term) | T+7 (Long-term) | Key Finding |
| :--- | :--- | :--- | :--- |
| **Linear MAE** | $\approx 1.135^{\circ}C$ | $\approx 1.315^{\circ}C$ | Highest error across the board. |
| **LGBM RMSE** | $\approx 1.48^{\circ}C$ | $\approx 1.64^{\circ}C$ | Best accuracy for the very short-term (T+1). |
| **XGBoost RMSE** | $\approx 1.485^{\circ}C$ | **$\approx 1.62^{\circ}C$** | **Best accuracy for the long-term (T+7)**, showing the lowest error. |

**Conclusion:** For practical forecasting, **XGBoost** is the best choice for the entire week, exhibiting the lowest error at the crucial T+7 horizon.

2. Model Fit (Test R²)

This metric measures the model's ability to explain the variance in temperature. **Higher values are better** and a flatter line indicates better stability.

![alt text](hourly/inference_results/compare_ALL_MODELS_Test_R2.png)

| Model | T+1 ($R^2$) | T+7 ($R^2$) | Stability Drop (T+1 to T+7) |
| :--- | :--- | :--- | :--- |
| **Linear** | $\approx 0.743$ | $\approx 0.67$ | **Sharpest decline (loss of $\approx 7.3$ points)**. |
| **LightGBM** | $\approx 0.755$ | $\approx 0.70$ | Maintains stability, dropping to $70\%$. |
| **XGBoost** | $\approx 0.75$ | **$\approx 0.705$** | **Highest $R^2$ at T+7**, demonstrating superior long-term stability. |

**Conclusion:** Both tree models maintain an $R^2$ of over $70\%$ at T+7, proving their robustness in modeling complex, non-linear hourly data. **XGBoost** is marginally more stable in the long run.

3. Overfitting Analysis (Train-Test RMSE Gap)

![alt text](hourly/inference_results/compare_ALL_MODELS_Overfitting_Gap.png)

This metric compares performance on the training and test sets. **Lower values are better**, indicating better generalization.

| Model | T+2 (Peak Overfitting) | T+7 (Generalization) | Overfitting Insight |
| :--- | :--- | :--- | :--- |
| **Linear** | $\approx 4.5\%$ | $\approx 3.5\%$ | **Lowest overall overfitting**, suggesting the best generalization ability despite the low accuracy. |
| **LightGBM** | **$\approx 10.5\%$** | $\approx 5.7\%$ | High initial overfitting, but the gap narrows significantly. |
| **XGBoost** | $\approx 10.5\%$ | **$\approx 4.8\%$** | High initial overfitting, but the gap reduces to the lowest among tree models at T+7. |

**Conclusion:** The **Linear Model** has the best generalization, but the high **accuracy** of the tree models justifies their higher overfitting gap. **XGBoost** manages to reduce its overfitting gap more effectively than LightGBM in the long term.

### 9.3. Final Conclusion (Step 8)

1.  **Hourly Winner:** For hourly data, **Tree-Based Models (LGBM/XGB) are superior** to Linear models in accuracy and stability (R²). The best choice depends on the horizon: **LightGBM for short-term (T+1 to T+6)** and **XGBoost for long-term (T+7)**.
2.  **Overall Winner (Daily vs. Hourly):**
      * **For pure *Accuracy* (lowest RMSE):** The **Daily Linear Model** is still the champion (Test RMSE T+1 of **0.828** beats the Hourly LGBM's \~1.48).
      * **For long-term *Stability* (best R²):** The **Hourly Tree Models** are the winners. Their T+7 R² of **\~70%** is far more reliable than the Daily model's T+7 R² of **43%**.

-----

## 🚀 10. Step 9: Deployment with ONNX

The final step involves optimizing our models for production deployment using ONNX (Open Neural Network Exchange). The goal is to convert our 7 trained Scikit-learn models from `.pkl` format to the optimized `.onnx` format.

### 10.1. Understanding ONNX and Its Value

#### A. What is ONNX?

**ONNX (Open Neural Network Exchange)** is an **open format** designed to represent machine learning models. It acts as an **interchange language**, allowing models trained in one framework (like Scikit-learn, PyTorch, or TensorFlow) to be converted and run efficiently on different platforms and devices (e.g., servers, edge devices, web services) using a highly optimized inference engine like ONNX Runtime.

#### B. Why Use ONNX for Deployment Efficiency?

ONNX should be used when maximizing **deployment efficiency** is critical:

1.  **Inference Speed (Low Latency):** Essential for applications requiring real-time, high-volume data processing.
2.  **Cross-Platform Compatibility:** Allows models to be deployed uniformly across various environments (e.g., Linux, Windows, Android) regardless of the original training setup.
3.  **Hardware Acceleration:** Enables seamless utilization of specialized hardware accelerators (GPU, FPGAs) via ONNX Runtime without custom code changes for each hardware type.
4.  **Model Optimization:** The ONNX Runtime can perform graph optimizations (e.g., node fusion) to further enhance computational efficiency.

---

### 10.2. Applying ONNX to Scikit-learn Models

In this project, we have 7 small Scikit-learn models (one for each forecast day) that are called frequently. Optimizing their inference speed is crucial, especially for an hourly forecasting system.

#### A. Rationale for Converting Scikit-learn to ONNX

Standard Scikit-learn inference is often hindered by the **Python Global Interpreter Lock (GIL)**. Converting the models to ONNX and using the **ONNX Runtime (ORT)**—which is written in high-performance C/C++—provides significant benefits:

* **Bypassing Python GIL:** ORT frees the model from Python's performance bottlenecks.
* **Leveraging Optimized Backends:** ORT utilizes highly efficient mathematical libraries and hardware-specific optimizations (like MKL or CUDA).
* **Simplified Deployment:** Ensures a consistent, fast, and dependency-light inference mechanism across production environments.

#### B. Final Inference Benchmark Results 🏆

The benchmark comparing the native Scikit-learn model with the converted ONNX model validates this approach:

| Method | Time\_ms\_per\_batch | Speedup\_vs\_Sklearn | Analysis |
| :--- | :--- | :--- | :--- |
| **Sklearn (CPU)** | **$0.9892 \text{ ms}$** | $1.0000$ | Baseline performance of the native Python model. |
| **ONNX (GPU)** | $0.0389 \text{ ms}$ | $\approx 25.4\text{x}$ | GPU acceleration is comparable to CPU, likely due to the small model size not justifying the overhead of data transfer to the GPU. |
| **ONNX (CPU)** | **$0.0383 \text{ ms}$** | **$\approx 25.8\text{x}$** | **Fastest performance!** Demonstrates the power of ONNX Runtime's optimization on the CPU, overcoming Python/Scikit-learn bottlenecks. |



**Conclusion:** Converting our Scikit-learn models to ONNX and utilizing the ONNX Runtime resulted in an astonishing **$\approx 25.8$ times speed increase** in inference time compared to the native Python implementation. This ensures the forecasting system can handle hourly prediction requests with extremely low latency in a production environment.




