# fine_tuning_linear.py (IMPORT FIX)
import os
import pandas as pd
import numpy as np
import yaml
import optuna
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from clearml import Task
import config
from feature_engineering import create_feature_pipeline # Import new pipeline
# DO NOT IMPORT FROM TRAIN_LINEAR ANYMORE

# ✅ ALIGN FUNCTION (COMPLEX) IS DEFINED HERE
def align_data_for_tuning(X_raw, y_raw, pipeline, scaler, fit_transform=False):
    """
    Standard align function: Run pipeline, scale, and join with y to dropna
    """
    # 1. Run pipeline (Input is X_raw, output is X_feat)
    if fit_transform:
        print("Fit/Transforming features...")
        X_feat = pipeline.fit_transform(X_raw)
        print("Fit/Transforming scaler...")
        X_scaled = scaler.fit_transform(X_feat)
    else:
        print("Transforming features...")
        X_feat = pipeline.transform(X_raw)
        print("Transforming scaler...")
        X_scaled = scaler.transform(X_feat)
    
    # 2. Align y
    y_aligned = y_raw.copy()
    y_aligned.index = X_feat.index 
    
    # 3. Repack to dropna
    y_df = pd.DataFrame(y_aligned)
    X_df = pd.DataFrame(X_scaled, index=X_feat.index, columns=X_feat.columns) 
    
    combined = pd.concat([y_df, X_df], axis=1)
    combined_clean = combined.dropna()
    
    y_final = combined_clean[y_aligned.name]
    X_final = combined_clean.drop(columns=[y_aligned.name])
    
    return X_final, y_final

def load_data_for_tuning(target_name):
    """Load Train and Val, but DO NOT merge"""
    print(f"🔍 Loading data for Optuna (Train/Val split) for {target_name}...")
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    
    # Assign index
    train_df['datetime'] = pd.to_datetime(train_df['datetime'])
    train_df = train_df.set_index('datetime', drop=False)
    val_df['datetime'] = pd.to_datetime(val_df['datetime'])
    val_df = val_df.set_index('datetime', drop=False)

    y_train_raw = train_df[target_name]
    X_train_raw = train_df.copy()
    
    y_val_raw = val_df[target_name]
    X_val_raw = val_df.copy()
    
    feature_pipeline_fit = create_feature_pipeline()
    scaler_fit = RobustScaler()

    # Align Train
    print("Aligning Train data...")
    # ✅ FIX: Call align_data_for_tuning (defined above)
    # ✅ FIX: Removed 'is_train'
    X_train_final, y_train_final = align_data_for_tuning(
        X_train_raw, y_train_raw, 
        feature_pipeline_fit, scaler_fit, fit_transform=True
    )
    
    # Align Val
    print("Aligning Val data...")
    X_val_final, y_val_final = align_data_for_tuning(
        X_val_raw, y_val_raw, 
        feature_pipeline_fit, scaler_fit, fit_transform=False
    )
    
    print(f"Train shapes: X={X_train_final.shape}, y={y_train_final.shape}")
    print(f"Val shapes: X={X_val_final.shape}, y={y_val_final.shape}")
    
    return X_train_final, y_train_final, X_val_final, y_val_final

def linear_objective(trial, X_train, y_train, X_val, y_val):
    ranges = config.LINEAR_PARAM_RANGES
    
    # 1. Chọn Model
    model_type = trial.suggest_categorical("model_type", ranges['model_type'])
    
    # ✅ SỬA: CHỈ LẤY PARAMS KHI CẦN
    if model_type == 'LinearRegression':
        model = LinearRegression(n_jobs=-1)
    
    else:
        # 2. Chọn Alpha (Sức mạnh Regularization)
        alpha = trial.suggest_float("alpha", *ranges['alpha'], log=True)

        if model_type == 'Ridge':
            model = Ridge(alpha=alpha, random_state=42)
            
        elif model_type == 'Lasso':
            model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
            
        elif model_type == 'ElasticNet':
            l1_ratio = trial.suggest_float("l1_ratio", *ranges['l1_ratio'])
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)

    # Fit
    model.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    trial.set_user_attr("val_rmse", float(val_rmse))
    
    return val_rmse

def run_optuna_search_linear():
    task = Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        task_name=config.CLEARML_TASK_NAME,
        tags=["Optuna", "LinearRegression", "Multi-Horizon"]
    )
    
    all_best_params = {}
    all_best_scores = {}
    all_best_details = {}

    for target_name in config.TARGET_FORECAST_COLS:
        print("\n" + "="*30)
        print(f"🎯 Tuning for: {target_name}")
        print("="*30)
    
        # Load data for this target
        X_train_final, y_train_final, X_val_final, y_val_final = load_data_for_tuning(target_name)
        
        print(f"🔍 Starting Optuna (Linear Models)...")
        
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: linear_objective(trial, X_train_final, y_train_final, X_val_final, y_val_final),
            n_trials=config.OPTUNA_TRIALS,
            show_progress_bar=True
        )
        
        best_trial = study.best_trial
        best_params = best_trial.params
        val_rmse = best_trial.user_attrs.get("val_rmse", 0)

        all_best_params[target_name] = best_params
        all_best_scores[target_name] = float(val_rmse)
        all_best_details[target_name] = {"val_rmse": float(val_rmse)}

        print(f"\n🏆 BEST RESULTS FOR {target_name}:")
        print(f"   Val RMSE:      {val_rmse:.4f}")
        print(f"   Best Model:    {best_params.get('model_type')}")
        if 'alpha' in best_params:
            print(f"   Best Alpha:    {best_params.get('alpha'):.4f}")
        if 'l1_ratio' in best_params:
            print(f"   Best L1 Ratio: {best_params.get('l1_ratio'):.4f}")
    
    # Save results
    output = {
        "best_params": all_best_params,
        "best_scores": all_best_scores,
        "details": all_best_details,
        "config": {
            "n_trials": config.OPTUNA_TRIALS,
            "search_space": config.LINEAR_PARAM_RANGES
        }
    }
    output_path = os.path.join(config.MODEL_DIR, config.OPTUNA_RESULTS_YAML)
    with open(output_path, "w") as f:
        yaml.dump(output, f, sort_keys=False)
    
    print(f"\n✅ OPTUNA LINEAR SEARCH COMPLETE.")
    print(f"📁 Best params saved to: {output_path}")
    print(f"\n🚀 NEXT STEP: Run 'python train_linear.py'")
    task.close()

if __name__ == "__main__":
    run_optuna_search_linear()