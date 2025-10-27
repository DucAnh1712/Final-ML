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
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from clearml import Task

# Import from other files
import config
from feature_engineering import create_feature_pipeline

def load_features_for_tuning(target_col):
    """Load created features for Optuna tuning."""
    print("🔍 Loading feature data for tuning...")
    train = pd.read_csv(os.path.join(config.FEATURE_DIR, "feature_train.csv"))
    val = pd.read_csv(os.path.join(config.FEATURE_DIR, "feature_val.csv"))

    # Concat train and val for Optuna's TimeSeriesSplit
    full_train_df = pd.concat([train, val], ignore_index=True)
    
    # Drop non-feature columns
    drop_cols = [target_col, 'datetime']
    features_df = full_train_df.drop(columns=drop_cols, errors='ignore')
    target_s = full_train_df[target_col]

    # Drop remaining object columns
    obj_cols = features_df.select_dtypes(include=['object']).columns
    if not obj_cols.empty:
        print(f"⚠️ Dropping object columns: {list(obj_cols)}")
        features_df = features_df.drop(columns=obj_cols)

    return features_df, target_s

def xgb_objective(trial, X, y):
    """Objective function for Optuna (Step 5)."""
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    rmse_scores = []

    # Define hyperparameters to tune
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
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

        # This pipeline only includes Scaler and Model
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("xgb", XGBRegressor(**params))
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

    return np.mean(rmse_scores)

def main():
    """Main pipeline: Tune -> Create Final Pipeline -> Retrain -> Save."""
    
    # 1. Initialize ClearML (Step 5)
    task = Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        task_name=config.CLEARML_TASK_NAME,
        tags=["Optuna", "XGBoost", "Daily"]
    )
    
    # 2. Load data (for tuning only)
    X_tune, y_tune = load_features_for_tuning(target_col=config.TARGET_COL)

    # 3. Run Optuna
    print(f"🚀 Starting Optuna tuning ({config.OPTUNA_TRIALS} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: xgb_objective(trial, X_tune, y_tune), n_trials=config.OPTUNA_TRIALS)
    
    best_params = study.best_params
    print(f"🏆 Best Params found: {best_params}")
    
    # Log best params to ClearML
    task.connect(best_params, name='Best Hyperparameters')
    task.get_logger().report_scalar("best_rmse", "RMSE", value=study.best_value, iteration=0)

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

    # Fit the complete pipeline
    production_pipeline.fit(X_train_full, y_train_full)

    # 6. SAVE PIPELINE
    model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    joblib.dump(production_pipeline, model_path)
    print(f"✅ Production pipeline saved to: {model_path}")
    
    # ======================================================
    # 7. SAVE TO ONNX FORMAT (STEP 9) - ĐÃ SỬA LẠI
    # ======================================================
    print("🛠️ Creating ONNX-convertible components (Scaler + Model)...")

    # 1. Tạo và huấn luyện Scaler
    scaler = RobustScaler() 
    X_train_full_feat, y_train_full_feat = load_features_for_tuning(config.TARGET_COL)
    scaler.fit(X_train_full_feat)
    
    # 2. Áp dụng Scaler
    X_train_scaled = scaler.transform(X_train_full_feat)

    # 3. Tạo và huấn luyện Model
    model_xgb = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    model_xgb.fit(X_train_scaled, y_train_full_feat)

    # 4. LƯU 2 FILE RIÊNG BIỆT
    # 4a. Lưu Scaler bằng joblib
    scaler_path = os.path.join(config.MODEL_DIR, "scaler_for_onnx.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ ONNX Scaler saved to: {scaler_path}")
    
    # 4b. Lưu XGBoost bằng .save_model (JSON) để tránh lỗi pickle
    model_json_path = os.path.join(config.MODEL_DIR, "model_for_onnx.json")
    model_xgb.save_model(model_json_path)
    print(f"✅ ONNX XGBoost Model saved to: {model_json_path}")
    # ======================================================

    # Save best_params.yaml
    params_path = os.path.join(config.MODEL_DIR, "best_params.yaml")
    with open(params_path, "w") as f:
        yaml.dump(best_params, f)
    
    task.close()
    print("🎉 Training complete!")

if __name__ == "__main__":
    main()