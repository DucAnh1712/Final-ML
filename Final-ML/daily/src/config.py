# config.py
import os

# ======================================================
# 1. ROOT PATH DEFINITION
# ======================================================
# Giả định code nằm trong thư mục /src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ======================================================
# 2. SUBDIRECTORY DEFINITIONS
# ======================================================
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
FEATURE_DIR = os.path.join(PROJECT_ROOT, 'feature_data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'inference_results')

# Tự động tạo thư mục
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# 3. DATA PARAMETERS
# ======================================================
RAW_FILE_NAME = "HCMWeatherDaily.xlsx" 
TARGET_COL = "temp" # Biến mục tiêu (sẽ bị xóa khỏi features)

# ======================================================
# 4. DATA SPLIT PARAMETERS
# ======================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (tự động)

# ======================================================
# 5. FEATURE ENGINEERING PARAMETERS (ĐÃ CẢI THIỆN)
# ======================================================
# Các cột thô dùng để TẠO XU HƯỚNG
# (Sau đó chúng sẽ bị xóa bởi DropRawFeatures)
LAG_COLS = [
    "dew", "humidity", "precip", "windspeed", 
    "sealevelpressure", "cloudcover", "solarradiation", 
    "uvindex", "is_rain", "is_cloudy", "is_clear"
]

# === YÊU CẦU: CHỈ DÙNG XU HƯỚNG ===
# Tắt LAGS bằng cách để rỗng
LAGS = []

# Mở rộng WINDOWS để có nhiều xu hướng hơn ("Cải thiện mô hình")
WINDOWS = [3, 7, 14, 21, 30]
# ===================================

# ======================================================
# 6. TRAINING PARAMETERS
# ======================================================
MODEL_NAME = "hcm_temp_pipeline.pkl"

# Tăng số lần thử để "chạy nát" Optuna
OPTUNA_TRIALS = 200 # (Bạn có thể tăng lên 300, 500)
CV_SPLITS = 5

# ======================================================
# 7. CLEARML PARAMETERS
# ======================================================
CLEARML_PROJECT_NAME = "HCM Weather Forecasting"
CLEARML_TASK_NAME = "XGBoost Optuna Tuning (Rolling Only)"