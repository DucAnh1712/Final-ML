# train.py
import os
import pandas as pd
import numpy as np
import joblib
import yaml
import optuna
# Xóa import Pruning để tránh lỗi
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
# from clearml import Task # <-- Comment/Xóa nếu không dùng

# Import from other files
import config
from feature_engineering import create_feature_pipeline

# ======================================================
# HÀM NÀY ĐÃ ĐƯỢC SỬA LẠI HOÀN TOÀN ĐỂ FIX LỖI DATA ALIGNMENT
# ======================================================
def load_features_for_tuning(target_col):
    """
    Tải features (X) từ feature_data/ và target (y) từ processed_data/
    để đảm bảo không có data leakage và dữ liệu được đồng bộ.
    """
    print("🔍 Loading aligned data for tuning (X from features, y from processed)...")
    
    # 1. Tải FEATURES (X) (Đã được tạo và dropna)
    train_feat_X_path = os.path.join(config.FEATURE_DIR, "feature_train.csv")
    val_feat_X_path = os.path.join(config.FEATURE_DIR, "feature_val.csv")
    
    if not os.path.exists(train_feat_X_path) or not os.path.exists(val_feat_X_path):
        raise FileNotFoundError(
            "Feature files not found. Please run feature_engineering.py first."
        )
        
    train_feat_X = pd.read_csv(train_feat_X_path)
    val_feat_X = pd.read_csv(val_feat_X_path)
    X_tune = pd.concat([train_feat_X, val_feat_X], ignore_index=True)

    # 2. Tải dữ liệu PROCESSED (Để lấy y)
    train_proc_path = os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv")
    val_proc_path = os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv")
    
    if not os.path.exists(train_proc_path) or not os.path.exists(val_proc_path):
        raise FileNotFoundError(
            "Processed data files not found. Please run data_processing.py first."
        )

    train_proc = pd.read_csv(train_proc_path)
    val_proc = pd.read_csv(val_proc_path)
    
    # 3. CĂN CHỈNH (ALIGN) y VỚI X
    # feature_engineering.py đã "dropna()" các hàng đầu tiên của train_feat_X
    
    original_train_len = len(train_proc)
    new_train_len = len(train_feat_X)
    rows_dropped_at_start = original_train_len - new_train_len
    
    if rows_dropped_at_start < 0:
        raise ValueError("Feature train set is larger than processed train set. Check logic.")
        
    print(f"Aligning data: {rows_dropped_at_start} rows were dropped from train set by feature_engineering (due to rolling windows).")

    # Lấy y (target) từ các file processed, BỎ ĐI các hàng đầu tiên
    y_train = train_proc[target_col].iloc[rows_dropped_at_start:]
    y_val = val_proc[target_col] # Tập val không bị dropna

    y_tune = pd.concat([y_train, y_val], ignore_index=True)

    # 4. Kiểm tra lần cuối
    if len(X_tune) != len(y_tune):
        raise ValueError(
            f"Data misalignment: X_tune has {len(X_tune)} rows, "
            f"but y_tune has {len(y_tune)} rows. "
        )
        
    obj_cols = X_tune.select_dtypes(include=['object']).columns
    if not obj_cols.empty:
        print(f"⚠️ Dropping object columns from X_tune: {list(obj_cols)}")
        X_tune = X_tune.drop(columns=obj_cols)

    return X_tune, y_tune
# ======================================================

# ======================================================
# HÀM NÀY ĐÃ SỬA LẠI (Bản đơn giản) ĐỂ TRÁNH LỖI PHIÊN BẢN
# ======================================================
def xgb_objective(trial, X, y):
    """Objective function for Optuna (Bản đơn giản, không Pruning)."""
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    rmse_scores = []

    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'gamma': trial.suggest_float("gamma", 0.0, 5.0),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'random_state': 42,
        'n_jobs': -1
    }

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("xgb", XGBRegressor(**params))
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

    return np.mean(rmse_scores)
# ======================================================

def main():
    """Main pipeline: Tune -> Create Final Pipeline -> Retrain -> Save."""
    
    # 1. Initialize ClearML (Step 5)
    # task = Task.init(...) # <-- Tắt nếu không dùng
    
    # 2. Load data (for tuning only)
    X_tune, y_tune = load_features_for_tuning(target_col=config.TARGET_COL)

    # 3. Run Optuna
    print(f"🚀 Starting Optuna tuning ({config.OPTUNA_TRIALS} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: xgb_objective(trial, X_tune, y_tune), n_trials=config.OPTUNA_TRIALS)
    
    best_params = study.best_params
    print(f"🏆 Best Params found: {best_params}")
    
    # Log best params to ClearML
    # task.connect(best_params, name='Best Hyperparameters') # <-- Tắt nếu không dùng
    # task.get_logger().report_scalar(...) # <-- Tắt nếu không dùng

    # 4. CREATE FINAL PRODUCTION PIPELINE
    print("🛠️ Creating final production pipeline...")
    production_pipeline = Pipeline([
        ('feature_engineering', create_feature_pipeline()),
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(**best_params, random_state=42, n_jobs=-1))
    ])

    # 5. RETRAIN ON FULL (TRAIN + VAL) DATASET
    print("🔄 Retraining pipeline on (Train + Val)...")
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    
    all_train_data = pd.concat([train_df, val_df], ignore_index=True)
    all_train_data["datetime"] = pd.to_datetime(all_train_data["datetime"])
    all_train_data = all_train_data.sort_values("datetime").reset_index(drop=True)

    X_train_full = all_train_data.drop(columns=[config.TARGET_COL], errors='ignore')
    y_train_full = all_train_data[config.TARGET_COL]

    production_pipeline.fit(X_train_full, y_train_full)

    # 6. SAVE PIPELINE
    model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    joblib.dump(production_pipeline, model_path)
    print(f"✅ Production pipeline saved to: {model_path}")
    
    # ======================================================
    # 7. SAVE TO ONNX FORMAT (STEP 9) - ĐÃ SỬA LẠI
    # ======================================================
    print("🛠️ Creating ONNX-convertible components (Scaler + Model)...")

    scaler = RobustScaler() 
    X_train_full_feat, y_train_full_feat = load_features_for_tuning(config.TARGET_COL)
    scaler.fit(X_train_full_feat)
    
    X_train_scaled = scaler.transform(X_train_full_feat)

    model_xgb = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    model_xgb.fit(X_train_scaled, y_train_full_feat)

    # 4. LƯU 2 FILE RIÊNG BIỆT
    scaler_path = os.path.join(config.MODEL_DIR, "scaler_for_onnx.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ ONNX Scaler saved to: {scaler_path}")
    
    model_json_path = os.path.join(config.MODEL_DIR, "model_for_onnx.json")
    model_xgb.save_model(model_json_path)
    print(f"✅ ONNX XGBoost Model saved to: {model_json_path}")
    # ======================================================

    # Save best_params.yaml
    params_path = os.path.join(config.MODEL_DIR, "best_params.yaml")
    with open(params_path, "w") as f:
        yaml.dump(best_params, f)
    
    # task.close() # <-- Tắt nếu không dùng
    print("🎉 Training complete!")

if __name__ == "__main__":
    main()