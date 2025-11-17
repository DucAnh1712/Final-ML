import os
import pandas as pd
import numpy as np
import yaml
import optuna
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from clearml import Task
import config_to_pred as config # Using config tailored for prediction process
from feature_engineering import create_feature_pipeline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# CUSTOM PURGED TIME SERIES SPLIT (ANTI-LEAKAGE)
# =============================================================================
class PurgedTimeSeriesSplit:
    """
    Time Series Cross-Validation with a gap (purge) between train and validation 
    sets to prevent data leakage.
    """
    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
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
# ALIGNMENT FUNCTION (HANDLES NAN FROM FEATURE ENGINEERING)
# =============================================================================
def align_data_for_tuning(X_raw, y_raw, pipeline, scaler, fit_transform=False):
    """
    Runs feature engineering, scales data, and aligns with the target (y) 
    to drop rows where the target is NaN.
    """
    # 1. Run pipeline and scale
    if fit_transform:
        X_feat = pipeline.fit_transform(X_raw)
        X_scaled = scaler.fit_transform(X_feat)
    else:
        X_feat = pipeline.transform(X_raw)
        X_scaled = scaler.transform(X_feat)
        
    # 2. Align y with X's index
    y_aligned = y_raw.copy()
    y_aligned.index = X_feat.index 
    
    # 3. Combine and drop NaN rows where the target is missing
    y_df = pd.DataFrame(y_aligned)
    X_df = pd.DataFrame(X_scaled, index=X_feat.index, columns=X_feat.columns) 
    
    combined = pd.concat([y_df, X_df], axis=1)
    combined_clean = combined.dropna(subset=[y_aligned.name]) 
    
    y_final = combined_clean[y_aligned.name]
    X_final = combined_clean.drop(columns=[y_aligned.name])
    
    return X_final, y_final

# =============================================================================
# DATA LOADING (INCLUDES CRITICAL TARGET DROPNA)
# =============================================================================
def load_data_for_tuning(target_name):
    """
    Loads combined train and validation data, processes 'datetime', sorts,
    and removes rows with NaN in the target column BEFORE CV.
    """
    print(f"🔍 Loading RAW data (Train + Val COMBINED) for {target_name}...")
    
    # 1. Load data
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    all_train_data = pd.concat([train_df, val_df], ignore_index=True)
    
    # 2. Process Datetime
    if 'datetime' not in all_train_data.columns:
          raise KeyError("❌ 'datetime' column not found in the CSV file.")
    print(f"   ...Using 'datetime' column as the time index")
    all_train_data['datetime'] = pd.to_datetime(all_train_data['datetime'])
    all_train_data = all_train_data.set_index('datetime', drop=False)
    all_train_data = all_train_data.sort_index()
    
    # 3. CRITICAL: Remove any rows with NaN values in the target column
    rows_before = len(all_train_data)
    all_train_data = all_train_data.dropna(subset=[target_name])
    rows_after = len(all_train_data)
    if rows_before > rows_after:
        print(f"   ⚠️ Removed {rows_before - rows_after} rows with NaN in the target column.")
        
    # 4. Extract X and y
    y_train_full_raw = all_train_data[target_name]
    X_train_full_raw = all_train_data.copy()
    
    print(f"✅ Combined RAW data shapes: X={X_train_full_raw.shape}, y={y_train_full_raw.shape}")
    print(f"📅 Date range: {X_train_full_raw.index.min()} → {X_train_full_raw.index.max()}")
    
    return X_train_full_raw, y_train_full_raw

# =============================================================================
# OBJECTIVE FUNCTION WITH EARLY STOPPING
# =============================================================================
def lightgbm_objective(trial, X_all_train_raw, y_all_train_raw):
    """
    Optuna objective function for LightGBM (with Early Stopping and Purged CV).
    """
    ranges = config.PREDICT_PARAM_RANGES
    
    # 1. Suggest hyperparameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', *ranges['learning_rate'], log=True),
        'max_depth': trial.suggest_int('max_depth', *ranges['max_depth']),
        'subsample': trial.suggest_float('subsample', *ranges['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *ranges['colsample_bytree']),
        'reg_alpha': trial.suggest_float('reg_alpha', *ranges['reg_alpha'], log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', *ranges['reg_lambda'], log=True),
        'num_leaves': trial.suggest_int('num_leaves', *ranges['num_leaves']),
        
        # --- Fixed parameters ---
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'n_estimators': 2000 # Set a large, fixed number
    }

    # 2. Setup Purged CV
    tscv = PurgedTimeSeriesSplit(
        n_splits=config.CV_N_SPLITS, 
        gap=config.CV_GAP_ROWS
    )
    fold_scores = []
    best_iterations = [] 
    
    # 3. Loop through folds
    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X_all_train_raw)):
        X_train_fold_raw = X_all_train_raw.iloc[train_idx]
        y_train_fold_raw = y_all_train_raw.iloc[train_idx]
        X_val_fold_raw = X_all_train_raw.iloc[val_idx]
        y_val_fold_raw = y_all_train_raw.iloc[val_idx]
        
        # Log first fold for verification
        if fold_num == 0:
            train_dates_col = X_train_fold_raw.index
            val_dates_col = X_val_fold_raw.index
            gap_duration = (val_dates_col.min() - train_dates_col.max())
            print(f"   ✅ Fold 1 verified:")
            print(f"       Train: {train_dates_col.min()} → {train_dates_col.max()}")
            print(f"       Gap:   {gap_duration} (approx {config.CV_GAP_DAYS} days)")
            print(f"       Val:   {val_dates_col.min()} → {val_dates_col.max()}")

        # 4. Create NEW pipeline and scaler for this fold
        feature_pipeline_fold = create_feature_pipeline()
        scaler_fold = RobustScaler()
        
        # 5. Fit/Transform on train fold ONLY
        X_train_fold, y_train_fold = align_data_for_tuning(
            X_train_fold_raw, y_train_fold_raw, 
            feature_pipeline_fold, scaler_fold, 
            fit_transform=True
        )
        # 6. Transform on val fold (using fitted pipeline/scaler)
        X_val_fold, y_val_fold = align_data_for_tuning(
            X_val_fold_raw, y_val_fold_raw, 
            feature_pipeline_fold, scaler_fold, 
            fit_transform=False
        )

        # 7. Train LightGBM model with early stopping
        model = lgb.LGBMRegressor(**params)
        
        if X_train_fold.empty or y_train_fold.empty:
            print(f"   ⚠️ Fold {fold_num+1} is empty. Skipping.")
            continue

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[lgb.early_stopping(100, verbose=False)] 
        )
        
        # Capture best iteration for final model training
        best_iteration = model.best_iteration_
        if best_iteration is None or best_iteration <= 0:
            # Fallback if early stopping didn't trigger
            best_iteration = params['n_estimators']
        best_iterations.append(best_iteration)
            
        # 8. Evaluate using the best iteration
        y_val_pred = model.predict(X_val_fold, num_iteration=best_iteration)
        val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        fold_scores.append(val_rmse)
    
    # 9. Return average RMSE and best iteration
    final_rmse = np.mean(fold_scores)
    avg_best_iteration = int(np.mean(best_iterations)) 
    
    trial.set_user_attr("val_rmse", float(final_rmse))
    trial.set_user_attr("avg_best_iteration", avg_best_iteration) 
    
    return final_rmse

# =============================================================================
# MAIN OPTUNA SEARCH FUNCTION
# =============================================================================
def run_optuna_search_lightgbm(): 
    """
    Main function to orchestrate the Optuna search for LightGBM models 
    across all specified target columns.
    """
    task = Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        task_name="Optuna LightGBM (Hourly)",
        tags=["Optuna", "LightGBM", "Multi-Horizon", "Purged-CV", "Hourly"] 
    )
    
    all_best_params = {}
    all_best_scores = {}
    all_best_details = {}

    # This loop automatically runs for all target horizons (e.g., t+1 to t+24)
    for target_name in config.TARGET_FORECAST_COLS:
        print("\n" + "="*80)
        print(f"🎯 TUNING LIGHTGBM FOR: {target_name}") 
        print("="*80)
    
        X_all_train_raw, y_all_train_raw = load_data_for_tuning(target_name)
        
        print(f"🔍 Starting Optuna search (LightGBM)...")
        print(f"   Strategy: {config.CV_N_SPLITS}-Fold Purged TimeSeriesSplit (Gap={config.CV_GAP_ROWS} rows/hours)")
        print(f"   Trials:   {config.OPTUNA_TRIALS}")
        print(f"   ⚠️   This will take time...\n")
        
        # Create and run Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: lightgbm_objective(trial, X_all_train_raw, y_all_train_raw), 
            n_trials=config.OPTUNA_TRIALS,
            show_progress_bar=True
        )
        
        # Extract best results
        best_trial = study.best_trial
        best_params = best_trial.params
        val_rmse = best_trial.user_attrs.get("val_rmse", 0)
        avg_best_iter = best_trial.user_attrs.get("avg_best_iteration", 0) 

        # Update n_estimators in the best params using the averaged best iteration
        best_params['n_estimators'] = avg_best_iter 

        all_best_params[target_name] = best_params
        all_best_scores[target_name] = float(val_rmse)
        all_best_details[target_name] = {
            "val_rmse": float(val_rmse), 
            "avg_best_iteration": avg_best_iter,
            "n_folds": config.CV_N_SPLITS, 
            "gap_rows": config.CV_GAP_ROWS
        }

        # Print results
        print(f"\n🏆 BEST LIGHTGBM RESULTS FOR {target_name}:") 
        print(f"   Avg Val RMSE: {val_rmse:.4f} (across {config.CV_N_SPLITS} folds)")
        print(f"   Best n_estimators (avg): {avg_best_iter}")
        print(f"   Best learning_rate: {best_params.get('learning_rate'):.6f}")

    # Save results to YAML
    output = {
        "best_params": all_best_params,
        "best_scores": all_best_scores,
        "details": all_best_details,
        "config": {
            "n_trials": config.OPTUNA_TRIALS,
            "cv_strategy": "PurgedTimeSeriesSplit",
            "n_splits": config.CV_N_SPLITS,
            "gap_rows": config.CV_GAP_ROWS,
            "search_space": config.PREDICT_PARAM_RANGES, 
            "leakage_safe": True
        }
    }
    
    output_path = os.path.join(config.MODEL_DIR, config.OPTUNA_RESULTS_LIGHTGBM_YAML)
    with open(output_path, "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)
    
    # Final summary
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