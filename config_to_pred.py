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
RAW_FILE_NAME = "HCMWeatherHourly.xlsx" 
TARGET_COL = "temp"

# ======================================================
# FORECAST HORIZONS (TÍNH BẰNG GIỜ)
# ======================================================
# ⬅️ THAY ĐỔI: Dự đoán 24 giờ tiếp theo (t+1, t+2, ..., t+24)
FORECAST_HORIZONS = list(range(1, 25)) 
TARGET_FORECAST_COLS = [f"target_t{h}" for h in FORECAST_HORIZONS]

# ======================================================
# CROSS-VALIDATION SETTINGS (CHO HOURLY)
# ======================================================
CV_N_SPLITS = 5          # Số folds
CV_GAP_DAYS = 7          # 7 ngày gap
CV_GAP_ROWS = CV_GAP_DAYS * 24  # 7 ngày * 24 giờ = 168 hàng gap

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

# ⬅️ Đã xóa tất cả các biến của Linear Model

# # XGBoost
# MODEL_NAME_XGBOOST = "model_xgboost_hourly.pkl"
# OPTUNA_RESULTS_XGBOOST_YAML = "optuna_best_params_xgboost_hourly.yaml"
# TRAIN_METRICS_XGBOOST_NAME = "train_metrics_xgboost_hourly.yaml"
# TEST_METRICS_XGBOOST_NAME = "test_metrics_xgboost_hourly.yaml"
# TEST_PREDS_XGBOOST_NAME = "test_predictions_xgboost_hourly.csv"

# LightGBM
MODEL_NAME_LIGHTGBM = "model_pred_hourly.pkl"
OPTUNA_RESULTS_LIGHTGBM_YAML = "optuna_best_params_pred_hourly.yaml"
TRAIN_METRICS_LIGHTGBM_NAME = "train_metrics_pred_hourly.yaml"
TEST_METRICS_LIGHTGBM_NAME = "test_metrics_pred_hourly.yaml"
TEST_PREDS_LIGHTGBM_NAME = "test_predictions_hourly.csv"

# ======================================================
# OPTUNA FINE-TUNING
# ======================================================
OPTUNA_TRIALS = 50 # 50 trials cho mỗi horizon (tổng cộng 50 * 24 = 1200)

# # ✅ Search space cho XGBoost (Đã xóa n_estimators)
# XGBOOST_PARAM_RANGES = {
#     # n_estimators được xác định bằng early stopping, không tune
#     'learning_rate': (1e-3, 1e-1),
#     'max_depth': (3, 10),
#     'subsample': (0.6, 1.0),
#     'colsample_bytree': (0.6, 1.0),
#     'reg_alpha': (1e-5, 1e2),
#     'reg_lambda': (1e-5, 1e2),
# }

# ✅ Search space cho LightGBM (Đã xóa n_estimators)
PREDICT_PARAM_RANGES = {
    # n_estimators được xác định bằng early stopping, không tune
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
CLEARML_PROJECT_NAME = "HCM Weather Forecasting (Hourly)"