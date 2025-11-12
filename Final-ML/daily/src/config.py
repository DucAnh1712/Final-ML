# config.py
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
# FEATURE_DIR = os.path.join(PROJECT_ROOT, 'feature_data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'inference_results')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# DATA PARAMETERS
# ======================================================
RAW_FILE_NAME = "HCMWeatherDaily.xlsx" 
TARGET_COL = "temp"
TARGET_FORECAST_COLS = ["target_t1", "target_t2", "target_t3", "target_t4", "target_t5", "target_t6", "target_t7"]

# ======================================================
# DATA SPLIT
# ======================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15 # 15% Val + 15% Test

# ======================================================
# MODEL FILENAMES
# ======================================================
PIPELINE_NAME = "feature_pipeline.pkl"
SCALER_NAME = "scaler.pkl"
MODEL_NAME = "model_linear.pkl"

# Result filenames
BENCHMARK_RESULTS_YAML = "benchmark_results.yaml"
TRAIN_METRICS_NAME = "train_metrics_linear.yaml"
OPTUNA_SCRIPT_NAME = "fine_tuning_linear.py"
OPTUNA_RESULTS_YAML = "optuna_best_params_linear.yaml"

# ======================================================
# OPTUNA FINE-TUNING
# ======================================================
OPTUNA_TRIALS = 300 

LINEAR_PARAM_RANGES = {
    # ✅ THÊM 'LinearRegression' VÀO ĐÂY
    'model_type': ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'],
    
    'alpha': (1e-3, 10.0), # (log=True)
    'l1_ratio': (0.1, 0.9)
}

# ======================================================
# CLEARML
# ======================================================
CLEARML_PROJECT_NAME = "HCM Weather Forecasting"
CLEARML_TASK_NAME = "LinearModel Fine-Tuning (7-day)"