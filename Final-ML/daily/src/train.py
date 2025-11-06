# train.py
import os
import pandas as pd
import numpy as np
import joblib
import yaml
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
# === THÊM METRICS ĐỂ CHECK OVERFITTING ===
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from clearml import Task 

# Import from other files
import config
from feature_engineering import create_feature_pipeline

# ======================================================
# === HÀM MỚI: TẢI DỮ LIỆU CHO NHIỀU TARGET ===
# ======================================================
def load_features_for_tuning_multi(target_cols_list):
    """
    Tải features (X) từ feature_data/
    Tải TẤT CẢ các targets (y) từ processed_data/
    """
    print("🔍 Loading aligned data for tuning (X from features, Y-dict from processed)...")
    
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

    # 2. Tải dữ liệu PROCESSED (Để lấy TẤT CẢ CÁC CỘT Y)
    train_proc_path = os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv")
    val_proc_path = os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv")
    
    if not os.path.exists(train_proc_path) or not os.path.exists(val_proc_path):
        raise FileNotFoundError(
            "Processed data files not found. Please run data_processing.py first."
        )

    train_proc = pd.read_csv(train_proc_path)
    val_proc = pd.read_csv(val_proc_path)
    
    # 3. CĂN CHỈNH (ALIGN) y VỚI X (Căn chỉnh các hàng bị drop ở ĐẦU)
    original_train_len = len(train_proc)
    new_train_len = len(train_feat_X)
    rows_dropped_at_start = original_train_len - new_train_len
    
    if rows_dropped_at_start < 0:
        raise ValueError("Feature train set is larger than processed train set. Check logic.")
        
    print(f"Aligning data: {rows_dropped_at_start} rows were dropped from train set by feature_engineering (due to rolling windows).")

    # Tạo một dictionary (từ điển) cho các Y
    y_tune_dict = {}
    
    for target_name in target_cols_list:
        # Lấy y (target) từ các file processed, BỎ ĐI các hàng đầu tiên
        y_train = train_proc[target_name].iloc[rows_dropped_at_start:]
        y_val = val_proc[target_name] # Tập val không bị dropna

        y_tune = pd.concat([y_train, y_val], ignore_index=True)
        
        # Lưu y (vẫn còn NaN ở cuối) vào dictionary
        y_tune_dict[target_name] = y_tune

    # 4. Kiểm tra
    if len(X_tune) != len(y_tune_dict[target_cols_list[0]]):
        raise ValueError("Data misalignment after start alignment. Check logic.")
        
    obj_cols = X_tune.select_dtypes(include=['object']).columns
    if not obj_cols.empty:
        print(f"⚠️ Dropping object columns from X_tune: {list(obj_cols)}")
        X_tune = X_tune.drop(columns=obj_cols)

    # Trả về X (đã căn chỉnh start) và Dict Y (đã căn chỉnh start, còn NaN ở end)
    return X_tune, y_tune_dict
# ======================================================

def xgb_objective(trial, X, y):
    """Objective function for Optuna (Bản đơn giản, không Pruning)."""
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    rmse_scores = []

    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
        'max_depth': trial.suggest_int("max_depth", 2, 6),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'gamma': trial.suggest_float("gamma", 0.0, 5.0),
        'min_child_weight': trial.suggest_int("min_child_weight", 5, 20),
        'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 5.0),
        'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 5.0),
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
# === HÀM MAIN ĐÃ ĐƯỢC VIẾT LẠI HOÀN TOÀN ===
# ======================================================
def main():
    """Main pipeline: Chạy 4 lần, 1 lần cho mỗi target."""
    
    # 1. Initialize ClearML (Step 5)
    task = Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        task_name=config.CLEARML_TASK_NAME,
        tags=["Optuna", "XGBoost", "Multi-Target", "RollingOnly"]
    )
    
    # 1. Tải dữ liệu (cho Optuna)
    X_tune_full, y_tune_dict_full = load_features_for_tuning_multi(
        config.TARGET_FORECAST_COLS
    )

    # Dictionary để lưu các params tốt nhất
    all_best_params = {}

    # === BỌC TRONG VÒNG LẶP ===
    for target_name in config.TARGET_FORECAST_COLS:
        print(f"\n🚀🚀🚀 Bắt đầu quy trình cho: {target_name} 🚀🚀🚀")
        
        y_tune = y_tune_dict_full[target_name]
        
        # === CĂN CHỈNH (ALIGN) END (Cho Optuna) ===
        # Quan trọng: Xóa các hàng NaN ở cuối (do shift) CỦA TARGET NÀY
        valid_indices_tune = y_tune.dropna().index
        X_tune_aligned = X_tune_full.loc[valid_indices_tune]
        y_tune_aligned = y_tune.loc[valid_indices_tune]
        
        print(f"Aligning data for {target_name}: Dropped {len(y_tune) - len(valid_indices_tune)} NaN rows from end.")
        
        # 3. Run Optuna (Cho target này)
        print(f"🚀 Starting Optuna tuning for {target_name}...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: xgb_objective(trial, X_tune_aligned, y_tune_aligned), 
            n_trials=config.OPTUNA_TRIALS
        )
        
        best_params = study.best_params
        all_best_params[target_name] = best_params
        print(f"🏆 Best Params found for {target_name}: {best_params}")
        
        # Log to ClearML
        task.connect(best_params, name=f'Best Hyperparameters ({target_name})')
        task.get_logger().report_scalar(f"best_rmse ({target_name})", "RMSE", value=study.best_value, iteration=0)

        # 4. Tạo Production Pipeline (Cho target này)
        print(f"🛠️ Creating final production pipeline for {target_name}...")
        production_pipeline = Pipeline([
            ('feature_engineering', create_feature_pipeline()),
            ('scaler', RobustScaler()),
            ('model', XGBRegressor(**best_params, random_state=42, n_jobs=-1))
        ])

        # 5. Retrain (Cho target này)
        print(f"🔄 Retraining pipeline on (Train + Val) for {target_name}...")
        train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
        val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
        all_train_data = pd.concat([train_df, val_df], ignore_index=True)
        all_train_data = all_train_data.sort_values("datetime").reset_index(drop=True)
        
        # Tách X và y (dùng đúng target_name)
        y_train_full = all_train_data[target_name]
        
        # X_train_full là TẤT CẢ, nhưng phải drop các cột target khác
        # và cột 'temp' gốc
        cols_to_drop_prod = config.TARGET_FORECAST_COLS + [config.TARGET_COL]
        X_train_full = all_train_data.drop(columns=cols_to_drop_prod, errors='ignore')
        
        # Căn chỉnh (Align) END cho Production
        valid_indices_prod = y_train_full.dropna().index
        X_train_full_aligned = X_train_full.loc[valid_indices_prod]
        y_train_full_aligned = y_train_full.loc[valid_indices_prod]

        production_pipeline.fit(X_train_full_aligned, y_train_full_aligned)

        # ======================================================
        # 5B. TÍNH VÀ LƯU TRAIN METRICS (ĐỂ CHECK OVERFITTING)
        # ======================================================
        print(f"📊 Calculating performance on the Training Set for {target_name}...")
        
        # Dự đoán trên X_train_full_aligned
        y_train_pred = production_pipeline.predict(X_train_full_aligned)
        
        # y_train_full_aligned là đáp án
        y_train_actual = y_train_full_aligned
        
        # Căn chỉnh (Align) START (Do pipeline tự dropna)
        if len(y_train_pred) < len(y_train_actual):
            rows_dropped_at_start_prod = len(y_train_actual) - len(y_train_pred)
            print(f"Aligning Train predictions: Dropping first {rows_dropped_at_start_prod} rows from actuals.")
            y_train_actual_aligned = y_train_actual.iloc[rows_dropped_at_start_prod:]
        else:
            y_train_actual_aligned = y_train_actual

        train_metrics = {
            "RMSE": np.sqrt(mean_squared_error(y_train_actual_aligned, y_train_pred)),
            "MAE": mean_absolute_error(y_train_actual_aligned, y_train_pred),
            "R2": r2_score(y_train_actual_aligned, y_train_pred)
        }
        
        print("\n--- Training Set Performance ---")
        print(f"   Train MAE ({target_name}): {train_metrics['MAE']:.4f}")
        print(f"   Train R2 ({target_name}) : {train_metrics['R2']:.4f}")
        print("----------------------------------\n")
        
        # Lưu file metrics
        metrics_path = os.path.join(config.OUTPUT_DIR, f"train_metrics_{target_name}.yaml")
        with open(metrics_path, "w") as f:
            yaml.dump(train_metrics, f, sort_keys=False)
        print(f"🧾 Training metrics saved to: {metrics_path}")
        # ======================================================

        # 6. Lưu Model (Cho target này)
        model_name = f"{target_name}_pipeline.pkl"
        model_path = os.path.join(config.MODEL_DIR, model_name)
        joblib.dump(production_pipeline, model_path)
        print(f"✅ Production pipeline saved to: {model_path}")
    
        # ======================================================
        # 7. SAVE TO ONNX FORMAT (STEP 9) (Cho target này)
        # ======================================================
        print(f"🛠️ Creating ONNX components for {target_name}...")
        
        scaler = RobustScaler() 
        scaler.fit(X_tune_aligned) # Dùng X_tune đã căn chỉnh cho target này
        
        X_train_scaled = scaler.transform(X_tune_aligned)

        model_xgb = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        model_xgb.fit(X_train_scaled, y_tune_aligned)

        # LƯU 2 FILE RIÊNG BIỆT (với tên target)
        scaler_name = f"scaler_{target_name}.pkl"
        model_json_name = f"model_{target_name}.json"
        
        scaler_path = os.path.join(config.MODEL_DIR, scaler_name)
        joblib.dump(scaler, scaler_path)
        print(f"✅ ONNX Scaler saved to: {scaler_path}")
        
        model_json_path = os.path.join(config.MODEL_DIR, model_json_name)
        model_xgb.save_model(model_json_path)
        print(f"✅ ONNX XGBoost Model saved to: {model_json_path}")
        # ======================================================

    # Save best_params.yaml
    params_path = os.path.join(config.MODEL_DIR, "all_best_params.yaml")
    with open(params_path, "w") as f:
        yaml.dump(all_best_params, f)
    
    task.close()
    print("\n🎉🎉🎉 Completed Training. 🎉🎉🎉")

if __name__ == "__main__":
    main()