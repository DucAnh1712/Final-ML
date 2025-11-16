# train_xgboost.py
import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.preprocessing import RobustScaler
import xgboost as xgb # ⬅️ THAY ĐỔI 1
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from clearml import Task 
import config
from feature_engineering import create_feature_pipeline

# ⬅️ THAY ĐỔI 2: Load đúng file params
def load_optuna_best_params_xgb():
    params_path = os.path.join(config.MODEL_DIR, config.OPTUNA_RESULTS_XGBOOST_YAML) # ⬅️ Đổi tên file
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"❌ {params_path} not found\n"
            f"Please run 'python optuna_search_xgboost.py' first!" # ⬅️ Đổi tên file
        )
    with open(params_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    print(f"✅ Loaded Optuna XGBoost params from: {params_path}")
    return data['best_params']

# (Hàm align_data_final giữ nguyên)
def align_data_final(X_feat_scaled_df, y_raw_series):
    y_aligned = y_raw_series.copy()
    y_aligned.index = X_feat_scaled_df.index 
    y_df = pd.DataFrame(y_aligned)
    combined = pd.concat([y_df, X_feat_scaled_df], axis=1)
    combined_clean = combined.dropna(subset=[y_aligned.name])
    y_final = combined_clean[y_aligned.name]
    X_final = combined_clean.drop(columns=[y_aligned.name])
    return X_final, y_final

# ⬅️ THAY ĐỔI 3: Tạo model XGBoost
def create_model_from_params_xgb(params):
    # Thêm các params mặc định
    model_params = params.copy()
    model_params.setdefault('random_state', 42)
    model_params.setdefault('n_jobs', -1)
    
    print(f"   Model: XGBoost (n_estimators={params.get('n_estimators')}, max_depth={params.get('max_depth')})")
    return xgb.XGBRegressor(**model_params)

def main():
    task = Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        #task_name=config.CLEARML_TASK_NAME + " (XGBoost Production)", # ⬅️ Đổi tên Task
        task_name="Optuna XGBoost (Hourly)", # ⬅️ Đổi tên
        tags=["Production", "XGBoost", "Multi-Horizon", "Hourly"] # ⬅️ Đổi Tags
    )
    
    try:
        all_best_params = load_optuna_best_params_xgb() # ⬅️ Gọi hàm mới
    except FileNotFoundError as e:
        print(str(e))
        return

    print(f"🚀 STARTING PRODUCTION TRAINING (XGBoost, Multi-Horizon, Hourly)") # ⬅️ Đổi tên
    print("="*70)

    # ======================================================
    # 1. LOAD DATA (Merge Train + Val)
    # (Copy y hệt từ train_linear.py, đã có logic xử lý datetime)
    # ======================================================
    print(f"📂 Loading data (Train+Val)...")
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    all_train_data = pd.concat([train_df, val_df], ignore_index=True)
    if 'datetime' not in all_train_data.columns:
         raise KeyError("❌ Không tìm thấy cột 'datetime' trong file CSV.")
    all_train_data['datetime'] = pd.to_datetime(all_train_data['datetime'])
    all_train_data = all_train_data.set_index('datetime', drop=False)
    all_train_data = all_train_data.sort_index()
    X_train_full = all_train_data.copy()

    # ======================================================
    # 2. FIT PIPELINE & SCALER (ONCE)
    # (Phần này giữ nguyên, nó sẽ đọc config.PIPELINE_NAME_HOURLY v.v.)
    # ======================================================
    feature_pipeline = create_feature_pipeline()
    scaler = RobustScaler()
    print("Fitting Feature Pipeline on 85% data...")
    X_feat_full = feature_pipeline.fit_transform(X_train_full)
    print("Fitting Scaler on 85% data...")
    scaler.fit(X_feat_full) 
    joblib.dump(feature_pipeline, os.path.join(config.MODEL_DIR, config.PIPELINE_NAME))
    joblib.dump(scaler, os.path.join(config.MODEL_DIR, config.SCALER_NAME))
    print(f"💾 Feature Pipeline saved to: {config.PIPELINE_NAME}")
    print(f"💾 Scaler saved to: {config.SCALER_NAME}")

    # ======================================================
    # 3. LOOP AND TRAIN EACH MODEL
    # ======================================================
    all_train_metrics = {}
    X_scaled_full = scaler.transform(X_feat_full)
    X_scaled_full_df = pd.DataFrame(X_scaled_full, index=X_feat_full.index, columns=X_feat_full.columns)

    for target_name in config.TARGET_FORECAST_COLS: 
        print("\n" + "="*30)
        print(f"🎯 Training XGBoost for: {target_name}") # ⬅️ Đổi tên
        print("="*30)
        
        y_train_full = all_train_data[target_name]

        X_final_train, y_final_train = align_data_final(
            X_scaled_full_df, y_train_full
        )
        print(f"📊 Final training data (aligned): X={X_final_train.shape}, y={y_final_train.shape}")
        
        # 5. FIT MODEL (FROM TUNE RESULTS)
        if target_name not in all_best_params:
            print(f"⚠️ Tuned params not found for {target_name}. Using default XGBoost.")
            model = xgb.XGBRegressor(n_jobs=-1, random_state=42) # ⬅️ Đổi default
        else:
            best_params = all_best_params[target_name]
            task.connect(best_params, name=f'Best Params ({target_name})')
            model = create_model_from_params_xgb(best_params) # ⬅️ Gọi hàm mới
        
        print(f"⏳ Training final {target_name} model...")
        model.fit(X_final_train, y_final_train) # ⬅️ Fit trên toàn bộ Train+Val
        print(f"✅ Training complete!")

        # 6. CALCULATE TRAIN METRICS
        y_train_pred = model.predict(X_final_train)
        train_metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(y_final_train, y_train_pred))),
            "MAE": float(mean_absolute_error(y_final_train, y_train_pred)),
            "R2": float(r2_score(y_final_train, y_train_pred))
        }
        all_train_metrics[target_name] = train_metrics
        print(f"   Train RMSE: {train_metrics['RMSE']:.4f}")

        # 7. SAVE MODEL (separate name for each target)
        # ⬅️ THAY ĐỔI 4: Dùng tên model XGB
        model_name = f"{target_name}_{config.MODEL_NAME_XGBOOST}" 
        model_path = os.path.join(config.MODEL_DIR, model_name)
        joblib.dump(model, model_path)
        print(f"💾 Model saved to: {model_path}")

    # ⬅️ THAY ĐỔI 5: Dùng tên file metrics XGB
    metrics_path = os.path.join(config.OUTPUT_DIR, config.TRAIN_METRICS_XGBOOST_NAME)
    with open(metrics_path, "w") as f:
        yaml.dump(all_train_metrics, f, sort_keys=False)
    print(f"\n💾 All train metrics saved to: {metrics_path}")
    
    print(f"\n🚀 NEXT STEP: Run 'python inference_xgboost.py'") # ⬅️ Đổi tên
    task.close()

if __name__ == "__main__":
    main()