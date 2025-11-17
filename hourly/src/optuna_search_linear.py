import os
import pandas as pd
import numpy as np
import yaml
import optuna
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
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
    (purge) between the training and validation sets to prevent data leakage.
    
    Parameters:
    -----------
    n_splits : int
        The number of splits (folds).
    gap : int
        The number of rows/steps to skip between the end of the training 
        set and the start of the validation set (purge window).
        Example: gap=7 means skipping 7 days/hours after training before 
        validation starts.
    
    Example:
    --------
    |-------- Train --------|   GAP (7 days)   |---- Val ----|
      Fit pipeline here         (⛔)            Evaluate here
    """
    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate the size of each fold
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Train: from start up to the split point
            train_end = (i + 1) * test_size
            
            # Gap: discard `gap` rows after train
            val_start = train_end + self.gap
            val_end = val_start + test_size
            
            # Ensure indices do not exceed the dataset boundary
            if val_end > n_samples:
                break
            
            train_indices = indices[:train_end]
            val_indices = indices[val_start:val_end]
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# =============================================================================
# ALIGNMENT FUNCTION (FIXED DROPNA LOGIC)
# =============================================================================
def align_data_for_tuning(X_raw, y_raw, pipeline, scaler, fit_transform=False):
    """
    Standard align function: Runs feature engineering pipeline, scales data, 
    and joins with the target (y) to handle NaNs resulting from feature creation.
    """
    # 1. Run pipeline
    if fit_transform:
        X_feat = pipeline.fit_transform(X_raw)
        X_scaled = scaler.fit_transform(X_feat)
    else:
        X_feat = pipeline.transform(X_raw)
        X_scaled = scaler.transform(X_feat)
    
    # 2. Align y with X's index
    y_aligned = y_raw.copy()
    y_aligned.index = X_feat.index 
    
    # 3. Combine and drop NaN
    y_df = pd.DataFrame(y_aligned)
    X_df = pd.DataFrame(X_scaled, index=X_feat.index, columns=X_feat.columns) 
    
    combined = pd.concat([y_df, X_df], axis=1)
    
    # ✅✅✅ FIX APPLIED HERE ✅✅✅
    # Only drop rows where the TARGET COLUMN (y) is NaN
    # (We assume X_df is clean of NaNs due to proper pipeline handling)
    combined_clean = combined.dropna(subset=[y_aligned.name]) 
    
    y_final = combined_clean[y_aligned.name]
    X_final = combined_clean.drop(columns=[y_aligned.name])
    
    return X_final, y_final

# =============================================================================
# DATA LOADING (INCLUDES CRITICAL DROPNA STEP)
# =============================================================================
def load_data_for_tuning(target_name):
    """
    Loads combined train and validation data, processes 'datetime', sorts,
    and removes rows with NaN in the target column BEFORE CV.
    """
    print(f"🔍 Loading RAW data (Train + Val COMBINED) for {target_name}...")
    
    # 1. Load CSVs
    train_df = pd.read_csv(
        os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv")
    )
    val_df = pd.read_csv(
        os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv")
    )
    
    # 2. Concatenate
    all_train_data = pd.concat([train_df, val_df], ignore_index=True)
    
    # 3. Process Datetime
    if 'datetime' not in all_train_data.columns:
          raise KeyError("❌ 'datetime' column not found in the CSV file.")
    
    print(f"   ...Using 'datetime' column as the time index")
    all_train_data['datetime'] = pd.to_datetime(all_train_data['datetime'])
    all_train_data = all_train_data.set_index('datetime', drop=False)

    # 4. Sort (Crucial for CV)
    all_train_data = all_train_data.sort_index()

    # 5. ✅✅✅ CRITICAL FIX ✅✅✅
    # Remove any rows with NaN values in the target column
    rows_before = len(all_train_data)
    all_train_data = all_train_data.dropna(subset=[target_name])
    rows_after = len(all_train_data)
    if rows_before > rows_after:
        print(f"   ⚠️ Removed {rows_before - rows_after} rows with NaN in the target column.")

    # 6. Extract X and y
    y_train_full_raw = all_train_data[target_name]
    X_train_full_raw = all_train_data.copy()
    
    print(f"✅ Combined RAW data shapes: X={X_train_full_raw.shape}, y={y_train_full_raw.shape}")
    print(f"📅 Date range: {X_train_full_raw.index.min()} → {X_train_full_raw.index.max()}")
    
    return X_train_full_raw, y_train_full_raw

# =============================================================================
# OPTUNA OBJECTIVE FUNCTION (WITH PURGED CV)
# =============================================================================
def linear_objective(trial, X_all_train_raw, y_all_train_raw):
    """
    Optuna objective function for linear models using Purged TimeSeriesSplit.
    Ensures: unique pipelines/scalers per fold, purge gap, and preserved temporal order.
    Minimizes the average Root Mean Squared Error (RMSE).
    """
    ranges = config.LINEAR_PARAM_RANGES
    
    # 1. Suggest model type and hyperparameters
    model_type = trial.suggest_categorical("model_type", ranges['model_type'])
    
    if model_type == 'LinearRegression':
        model_params = {'n_jobs': -1}
    else:
        alpha = trial.suggest_float("alpha", *ranges['alpha'], log=True)
        model_params = {'alpha': alpha, 'random_state': 42}
        
        if model_type == 'Lasso':
            model_params['max_iter'] = 2000
        elif model_type == 'ElasticNet':
            model_params['l1_ratio'] = trial.suggest_float("l1_ratio", *ranges['l1_ratio'])
            model_params['max_iter'] = 2000

    # ✅ 2. Purged Time Series CV with gap
    tscv = PurgedTimeSeriesSplit(
        n_splits=config.CV_N_SPLITS, 
        gap=config.CV_GAP_ROWS
    )
    
    fold_scores = []
    
    # 3. Loop through folds
    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X_all_train_raw)):
        # ✅ Get raw data (NO reset index - preserve temporal order)
        X_train_fold_raw = X_all_train_raw.iloc[train_idx]
        y_train_fold_raw = y_all_train_raw.iloc[train_idx]
        X_val_fold_raw = X_all_train_raw.iloc[val_idx]
        y_val_fold_raw = y_all_train_raw.iloc[val_idx]
        
        # ✅ CRITICAL CHECK: Verify temporal order
        train_dates = X_train_fold_raw.index
        val_dates = X_val_fold_raw.index
        
        # Ensure training period ends before validation period starts
        assert train_dates.max() < val_dates.min(), \
            f"❌ FOLD {fold_num+1} LEAKAGE DETECTED! Train ends {train_dates.max()}, Val starts {val_dates.min()}"
        
        # Log first fold for verification
        if fold_num == 0:
            # Calculate gap in days based on datetime index
            gap_duration = (val_dates.min() - train_dates.max())
            print(f"  ✅ Fold 1 verified:")
            print(f"     Train: {train_dates.min()} → {train_dates.max()}")
            print(f"     Gap:   {gap_duration} (approx {config.CV_GAP_DAYS} days)")
            print(f"     Val:   {val_dates.min()} → {val_dates.max()}")

        # ✅ 4. Create NEW pipeline and scaler for this fold (no data leakage)
        feature_pipeline_fold = create_feature_pipeline()
        scaler_fold = RobustScaler()

        # ✅ 5. Fit/Transform on train fold ONLY
        X_train_fold, y_train_fold = align_data_for_tuning(
            X_train_fold_raw, y_train_fold_raw, 
            feature_pipeline_fold, scaler_fold, 
            fit_transform=True  # ← Fit on train
        )
        
        # ✅ 6. Transform on val fold (using fitted pipeline/scaler)
        X_val_fold, y_val_fold = align_data_for_tuning(
            X_val_fold_raw, y_val_fold_raw, 
            feature_pipeline_fold, scaler_fold, 
            fit_transform=False  # ← Only transform on val
        )

        # 7. Initialize and Train model
        if model_type == 'LinearRegression':
            model = LinearRegression(**model_params)
        elif model_type == 'Ridge':
            model = Ridge(**model_params)
        elif model_type == 'Lasso':
            model = Lasso(**model_params)
        elif model_type == 'ElasticNet':
            model = ElasticNet(**model_params)
        
        model.fit(X_train_fold, y_train_fold)
        
        # 8. Evaluate
        y_val_pred = model.predict(X_val_fold)
        val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        fold_scores.append(val_rmse)
    
    # 9. Return average RMSE across all folds
    final_rmse = np.mean(fold_scores)
    trial.set_user_attr("val_rmse", float(final_rmse))
    
    return final_rmse

# =============================================================================
# MAIN OPTUNA SEARCH FUNCTION (CONFIG FIXES APPLIED)
# =============================================================================
def run_optuna_search_linear():
    """
    Main function to orchestrate the Optuna search for linear models 
    across all specified target columns.
    """
    # Initialize ClearML task
    task = Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        # FIX 1: Manually set task name
        task_name="Optuna Linear (Hourly)", 
        tags=["Optuna", "LinearRegression", "Multi-Horizon", "Purged-CV", "Hourly"]
    )
    
    all_best_params = {}
    all_best_scores = {}
    all_best_details = {}

    # Loop through all forecast horizons (target columns)
    for target_name in config.TARGET_FORECAST_COLS:
        print("\n" + "="*80)
        print(f"🎯 TUNING FOR: {target_name}")
        print("="*80)
        
        # Load raw data
        X_all_train_raw, y_all_train_raw = load_data_for_tuning(target_name)
        
        print(f"🔍 Starting Optuna search...")
        # FIX 2: Print GAP_ROWS correctly
        print(f"   Strategy: {config.CV_N_SPLITS}-Fold Purged TimeSeriesSplit (Gap={config.CV_GAP_ROWS} rows/hours)")
        print(f"   Trials:   {config.OPTUNA_TRIALS}")
        print(f"   ⚠️  This will take time (re-fits pipelines in each fold)...\n")
        
        # Create and run Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: linear_objective(trial, X_all_train_raw, y_all_train_raw),
            n_trials=config.OPTUNA_TRIALS,
            show_progress_bar=True
        )
        
        # Extract best results
        best_trial = study.best_trial
        best_params = best_trial.params
        val_rmse = best_trial.user_attrs.get("val_rmse", 0)

        all_best_params[target_name] = best_params
        all_best_scores[target_name] = float(val_rmse)
        all_best_details[target_name] = {
            "val_rmse": float(val_rmse),
            "n_folds": config.CV_N_SPLITS,
            # FIX 3: Save GAP_ROWS correctly
            "gap_rows": config.CV_GAP_ROWS 
        }

        # Print results
        print(f"\n🏆 BEST RESULTS FOR {target_name}:")
        print(f"   Model Type:    {best_params.get('model_type')}")
        print(f"   Avg Val RMSE:  {val_rmse:.4f} (across {config.CV_N_SPLITS} folds)")
        if 'alpha' in best_params:
            print(f"   Best Alpha:    {best_params.get('alpha'):.6f}")
        if 'l1_ratio' in best_params:
            print(f"   Best L1 Ratio: {best_params.get('l1_ratio'):.4f}")
    
    # Save results to YAML
    output = {
        "best_params": all_best_params,
        "best_scores": all_best_scores,
        "details": all_best_details,
        "config": {
            "n_trials": config.OPTUNA_TRIALS,
            "cv_strategy": "PurgedTimeSeriesSplit",
            "n_splits": config.CV_N_SPLITS,
            # FIX 4: Save GAP_ROWS correctly
            "gap_rows": config.CV_GAP_ROWS, 
            "search_space": config.LINEAR_PARAM_RANGES,
            "leakage_safe": True
        }
    }
    
    # FIX 5: Use the correct output YAML file name
    output_path = os.path.join(config.MODEL_DIR, config.OPTUNA_RESULTS_LINEAR_YAML)
    with open(output_path, "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ OPTUNA LINEAR SEARCH COMPLETE (LEAKAGE-FREE)")
    print("="*80)
    print(f"📁 Best params saved to: {output_path}")
    print(f"\n📊 Summary of Best RMSE:")
    for target, score in all_best_scores.items():
        print(f"   {target}: {score:.4f}")
        
    # FIX 6: Update next-step file name
    print(f"\n🚀 NEXT STEP: Run 'python train_linear.py' to train final models")
    print("="*80)
    
    task.close()


if __name__ == "__main__":
    run_optuna_search_linear()