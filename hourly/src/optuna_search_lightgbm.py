# hourly/src/optuna_search_lightgbm.py
import os
import pandas as pd
import numpy as np
import yaml
import optuna
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from clearml import Task
import config
from feature_engineering import create_feature_pipeline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# CUSTOM PURGED TIME SERIES SPLIT (ANTI-LEAKAGE)
# =============================================================================
class PurgedTimeSeriesSplit:
    """
    Implements a Time Series Cross-Validation strategy with a gap 
    between the training and validation sets (purge) to prevent leakage.
    """
    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap
        
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        # Calculate test size based on the number of splits
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * test_size
            val_start = train_end + self.gap
            val_end = val_start + test_size
            
            if val_end > n_samples:
                break
                
            train_indices = indices[:train_end]
            val_indices = indices[val_start:val_end]
            yield train_indices, val_indices
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# =============================================================================
# ALIGNMENT FUNCTION
# =============================================================================
def align_data_for_tuning(X_raw, y_raw, pipeline, scaler, fit_transform=False):
    """
    Applies feature engineering and scaling, then aligns X and y 
    by dropping rows containing NaNs in the feature set resulting from 
    lag/rolling calculations.
    """
    if fit_transform:
        X_feat = pipeline.fit_transform(X_raw)
        X_scaled = scaler.fit_transform(X_feat)
    else:
        X_feat = pipeline.transform(X_raw)
        X_scaled = scaler.transform(X_feat)
        
    # Align index of y with X_feat (before scaling)
    y_aligned = y_raw.copy()
    y_aligned.index = X_feat.index 
    y_df = pd.DataFrame(y_aligned)
    
    # Convert scaled numpy array back to DataFrame with correct index and columns
    X_df = pd.DataFrame(X_scaled, index=X_feat.index, columns=X_feat.columns) 
    
    # Combine and drop rows where the target (y) is NaN
    combined = pd.concat([y_df, X_df], axis=1)
    combined_clean = combined.dropna(subset=[y_aligned.name]) 
    
    # Separate cleaned data
    y_final = combined_clean[y_aligned.name]
    X_final = combined_clean.drop(columns=[y_aligned.name])
    
    return X_final, y_final

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data_for_tuning(target_name):
    """
    Loads combined train and validation data, sets 'datetime' index, 
    sorts, and cleans the target column.
    """
    print(f"🔍 Loading RAW data (Train + Val COMBINED) for {target_name}...")
    
    # Load and combine the data
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    all_train_data = pd.concat([train_df, val_df], ignore_index=True)
    
    # Set datetime index
    if 'datetime' not in all_train_data.columns:
          raise KeyError("❌ 'datetime' column not found in the CSV file.")
    print(f"   ...Using 'datetime' column as time index")
    all_train_data['datetime'] = pd.to_datetime(all_train_data['datetime'])
    all_train_data = all_train_data.set_index('datetime', drop=False)
    all_train_data = all_train_data.sort_index()
    
    # Clean target column (y)
    rows_before = len(all_train_data)
    all_train_data = all_train_data.dropna(subset=[target_name])
    rows_after = len(all_train_data)
    
    if rows_before > rows_after:
        print(f"   ⚠️ Removed {rows_before - rows_after} rows with NaN in the target column.")
        
    y_train_full_raw = all_train_data[target_name]
    X_train_full_raw = all_train_data.copy()
    
    print(f"✅ Combined RAW data shapes: X={X_train_full_raw.shape}, y={y_train_full_raw.shape}")
    print(f"📅 Date range: {X_train_full_raw.index.min()} → {X_train_full_raw.index.max()}")
    
    return X_train_full_raw, y_train_full_raw

def lightgbm_objective(trial, X_all_train_raw, y_all_train_raw):
    """
    Optuna objective function for LightGBM hyperparameter search using Purged TimeSeriesSplit.
    Minimizes the average Root Mean Squared Error (RMSE) across all CV folds.
    """
    ranges = config.LIGHTGBM_PARAM_RANGES
    
    # Define hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', *ranges['n_estimators']),
        'learning_rate': trial.suggest_float('learning_rate', *ranges['learning_rate'], log=True),
        'max_depth': trial.suggest_int('max_depth', *ranges['max_depth']),
        'subsample': trial.suggest_float('subsample', *ranges['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *ranges['colsample_bytree']),
        'reg_alpha': trial.suggest_float('reg_alpha', *ranges['reg_alpha'], log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', *ranges['reg_lambda'], log=True),
        'num_leaves': trial.suggest_int('num_leaves', *ranges['num_leaves']),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Custom Time Series Cross-Validation
    tscv = PurgedTimeSeriesSplit(
        n_splits=config.CV_N_SPLITS, 
        gap=config.CV_GAP_ROWS
    )
    fold_scores = []
    
    # Loop through all CV folds
    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X_all_train_raw)):
        X_train_fold_raw = X_all_train_raw.iloc[train_idx]
        y_train_fold_raw = y_all_train_raw.iloc[train_idx]
        X_val_fold_raw = X_all_train_raw.iloc[val_idx]
        y_val_fold_raw = y_all_train_raw.iloc[val_idx]
        
        # Verify first fold's date gap (for logging/debugging)
        if fold_num == 0:
            train_dates_col = X_train_fold_raw['datetime'] 
            val_dates_col = X_val_fold_raw['datetime']
            gap_duration = (val_dates_col.min() - train_dates_col.max())
            print(f"   ✅ Fold 1 verified:")
            print(f"      Train: {train_dates_col.min()} → {train_dates_col.max()}")
            print(f"      Gap:   {gap_duration} (approx {config.CV_GAP_DAYS} days)")
            print(f"      Val:   {val_dates_col.min()} → {val_dates_col.max()}")

        # 1. Feature Engineering and Scaling (Fit on Train, Transform on Train/Val)
        feature_pipeline_fold = create_feature_pipeline()
        scaler_fold = RobustScaler()
        
        # Train data prep
        X_train_fold, y_train_fold = align_data_for_tuning(
            X_train_fold_raw, y_train_fold_raw, 
            feature_pipeline_fold, scaler_fold, 
            fit_transform=True
        )
        
        # Validation data prep (only transform)
        X_val_fold, y_val_fold = align_data_for_tuning(
            X_val_fold_raw, y_val_fold_raw, 
            feature_pipeline_fold, scaler_fold, 
            fit_transform=False
        )

        # 2. Model Training
        model = lgb.LGBMRegressor(**params)
        
        if X_train_fold.empty or y_train_fold.empty:
            print(f"   ⚠️ Fold {fold_num+1} is empty. Skipping.")
            continue

        model.fit(X_train_fold, y_train_fold)
        
        # 3. Prediction and Evaluation
        y_val_pred = model.predict(X_val_fold)
        val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        fold_scores.append(val_rmse)
    
    final_rmse = np.mean(fold_scores)
    
    # Report score back to Optuna trial
    trial.set_user_attr("val_rmse", float(final_rmse))
    
    return final_rmse

def run_optuna_search_lightgbm():
    """
    Main function to orchestrate the Optuna search for LightGBM models 
    across all specified target columns.
    """
    # Initialize ClearML task
    task = Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        task_name="Optuna LightGBM (Hourly)",
        tags=["Optuna", "LightGBM", "Multi-Horizon", "Purged-CV", "Hourly"] 
    )
    
    all_best_params = {}
    all_best_scores = {}
    all_best_details = {}

    # Iterate through all targets
    for target_name in config.TARGET_FORECAST_COLS:
        print("\n" + "="*80)
        print(f"🎯 TUNING LIGHTGBM FOR: {target_name}")
        print("="*80)
        
        X_all_train_raw, y_all_train_raw = load_data_for_tuning(target_name)
        
        print(f"🔍 Starting Optuna search (LightGBM)...")
        print(f"   Strategy: {config.CV_N_SPLITS}-Fold Purged TimeSeriesSplit (Gap={config.CV_GAP_ROWS} rows/hours)")
        print(f"   Trials:   {config.OPTUNA_TRIALS}")
        print(f"   ⚠️  This will take time...\n")
        
        # Create Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: lightgbm_objective(trial, X_all_train_raw, y_all_train_raw),
            n_trials=config.OPTUNA_TRIALS,
            show_progress_bar=True
        )
        
        # Save results
        best_trial = study.best_trial
        best_params = best_trial.params
        val_rmse = best_trial.user_attrs.get("val_rmse", 0)
        all_best_params[target_name] = best_params
        all_best_scores[target_name] = float(val_rmse)
        all_best_details[target_name] = {"val_rmse": float(val_rmse), "n_folds": config.CV_N_SPLITS, "gap_rows": config.CV_GAP_ROWS}

        print(f"\n🏆 BEST LIGHTGBM RESULTS FOR {target_name}:")
        print(f"   Avg Val RMSE: {val_rmse:.4f} (across {config.CV_N_SPLITS} folds)")
        print(f"   Best n_estimators: {best_params.get('n_estimators')}")
        print(f"   Best learning_rate: {best_params.get('learning_rate'):.6f}")

    # Final output structure
    output = {
        "best_params": all_best_params,
        "best_scores": all_best_scores,
        "details": all_best_details,
        "config": {
            "n_trials": config.OPTUNA_TRIALS,
            "cv_strategy": "PurgedTimeSeriesSplit",
            "n_splits": config.CV_N_SPLITS,
            "gap_rows": config.CV_GAP_ROWS,
            "search_space": config.LIGHTGBM_PARAM_RANGES,
            "leakage_safe": True
        }
    }
    
    # Save results to YAML file
    output_path = os.path.join(config.MODEL_DIR, config.OPTUNA_RESULTS_LIGHTGBM_YAML)
    with open(output_path, "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)
    
    print("\n" + "="*80)
    print("✅ OPTUNA LIGHTGBM SEARCH COMPLETE (LEAKAGE-FREE)")
    print("="*80)
    print(f"📁 Best params saved to: {output_path}")
    print(f"\n📊 Summary of Best RMSE:")
    for target, score in all_best_scores.items():
        print(f"   {target}: {score:.4f}")
    
    print(f"\n🚀 NEXT STEP: Run 'python train_lightgbm.py' to train final models")
    print("="*80)
    
    task.close()

if __name__ == "__main__":
    run_optuna_search_lightgbm()