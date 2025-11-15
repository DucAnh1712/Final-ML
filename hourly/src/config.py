import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'inference_results')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# DATA PARAMETERS
# ======================================================
RAW_FILE_NAME = "HCMWeatherHourly.xlsx" # ✅ Đã đổi sang Hourly
TARGET_COL = "temp"

# ======================================================
# FORECAST HORIZONS (TÍNH BẰNG GIỜ)
# ======================================================
# Dự đoán 7 ngày tới: 24h (T+1d), 48h (T+2d), ..., 168h (T+7d)
FORECAST_HORIZONS = [24, 48, 72, 96, 120, 144, 168]
TARGET_FORECAST_COLS = [f"target_t{h}" for h in FORECAST_HORIZONS]

# ======================================================
# CROSS-VALIDATION SETTINGS (CHO HOURLY)
# ======================================================
CV_N_SPLITS = 5          # Số folds
CV_GAP_DAYS = 7          # 7 ngày gap
CV_GAP_ROWS = CV_GAP_DAYS * 24  # ✅ QUAN TRỌNG: 7 ngày * 24 giờ = 168 hàng gap

# ======================================================
# DATA SPLIT
# ======================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15 # 15% Val + 15% Test

# ======================================================
# MODEL FILENAMES (CHO HOURLY)
# ======================================================
# Chung
PIPELINE_NAME = "feature_pipeline_hourly.pkl"
SCALER_NAME = "scaler_hourly.pkl"
BENCHMARK_RESULTS_YAML = "benchmark_results_hourly.yaml"

# Linear Models
MODEL_NAME_LINEAR = "model_linear_hourly.pkl"
OPTUNA_RESULTS_LINEAR_YAML = "optuna_best_params_linear_hourly.yaml"
TRAIN_METRICS_LINEAR_NAME = "train_metrics_linear_hourly.yaml"
TEST_METRICS_LINEAR_NAME = "test_metrics_linear_hourly.yaml" # ⬅️ THÊM DÒNG NÀY
TEST_PREDS_LINEAR_NAME = "test_predictions_linear_hourly.csv" # ⬅️ THÊM DÒNG NÀY

# XGBoost
MODEL_NAME_XGBOOST = "model_xgboost_hourly.pkl"
OPTUNA_RESULTS_XGBOOST_YAML = "optuna_best_params_xgboost_hourly.yaml"
TRAIN_METRICS_XGBOOST_NAME = "train_metrics_xgboost_hourly.yaml"
TEST_METRICS_XGBOOST_NAME = "test_metrics_xgboost_hourly.yaml" # ⬅️ THÊM DÒNG NÀY
TEST_PREDS_XGBOOST_NAME = "test_predictions_xgboost_hourly.csv" # ⬅️ THÊM DÒNG NÀY

# LightGBM
MODEL_NAME_LIGHTGBM = "model_lightgbm_hourly.pkl"
OPTUNA_RESULTS_LIGHTGBM_YAML = "optuna_best_params_lightgbm_hourly.yaml"
TRAIN_METRICS_LIGHTGBM_NAME = "train_metrics_lightgbm_hourly.yaml"
TEST_METRICS_LIGHTGBM_NAME = "test_metrics_lightgbm_hourly.yaml" # ⬅️ THÊM DÒNG NÀY
TEST_PREDS_LIGHTGBM_NAME = "test_predictions_lightgbm_hourly.csv" # ⬅️ THÊM DÒNG NÀY

# ======================================================
# OPTUNA FINE-TUNING
# ======================================================
# Giảm số trials vì dữ liệu hourly lớn hơn 24 lần
OPTUNA_TRIALS = 50 

# Search space cho Linear Models
LINEAR_PARAM_RANGES = {
    'model_type': ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'],
    'alpha': (1e-5, 1e2), # (log=True)
    'l1_ratio': (0.1, 0.9)
}

# ✅ THÊM: Search space cho XGBoost
XGBOOST_PARAM_RANGES = {
    'n_estimators': (200, 2000),
    'learning_rate': (1e-3, 1e-1),
    'max_depth': (3, 10),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (1e-5, 1e2),
    'reg_lambda': (1e-5, 1e2),
}

# ✅ THÊM: Search space cho LightGBM
LIGHTGBM_PARAM_RANGES = {
    'n_estimators': (200, 2000),
    'learning_rate': (1e-3, 1e-1),
    'max_depth': (3, 10),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (1e-5, 1e2),
    'reg_lambda': (1e-5, 1e2),
    'num_leaves': (20, 150)
}

# ======================================================
# CLEARML
# ======================================================
CLEARML_PROJECT_NAME = "HCM Weather Forecasting (Hourly)" # ✅ Đổi tên Project
# Tên Task sẽ được set trong mỗi file (ví dụ: Linear Tuning, XGB Tuning...)