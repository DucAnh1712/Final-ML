# config.py
import os

# ======================================================
# 1. ROOT PATH DEFINITION
# ======================================================
# Get the absolute path of the directory containing this config.py file
# Assuming structure: project_root/src/config.py
# -> PROJECT_ROOT will be project_root/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ======================================================
# 2. SUBDIRECTORY DEFINITIONS
# ======================================================
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
FEATURE_DIR = os.path.join(PROJECT_ROOT, 'feature_data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'inference_results')

# Automatically create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📂 Project Root: {PROJECT_ROOT}")

# ======================================================
# 3. DATA PARAMETERS
# ======================================================
# Change this to your data file (e.g., HanoiWeather.xlsx)
RAW_FILE_NAME = "HCMWeatherDaily.xlsx" 
TARGET_COL = "temp" # Target column to predict

# ======================================================
# 4. DATA SPLIT PARAMETERS
# ======================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO (automatic)

# ======================================================
# 5. FEATURE ENGINEERING PARAMETERS
# ======================================================
# Columns for creating lag/rolling features
LAG_COLS = ["temp", "humidity", "precip", "windspeed", "sealevelpressure"]
# Time intervals (days) for lags
LAGS = [1, 2, 3, 7, 14]
# Rolling windows (days)
WINDOWS = [3, 7, 14]
# Max lag (used for dropping NaNs in train set)
MAX_LAG = max(max(LAGS), max(WINDOWS))

# ======================================================
# 6. TRAINING PARAMETERS
# ======================================================
# Final model filename
MODEL_NAME = "hcm_temp_pipeline.pkl"
# Number of Optuna trials
OPTUNA_TRIALS = 50
# Number of folds for TimeSeriesSplit
CV_SPLITS = 5

# ======================================================
# 7. CLEARML PARAMETERS (Step 5)
# ======================================================
CLEARML_PROJECT_NAME = "HCM Weather Forecasting"
CLEARML_TASK_NAME = "XGBoost Optuna Tuning"