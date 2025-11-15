# optuna_search_linear.py (V4 - LEAKAGE-FREE WITH GAP)
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


# =============================================================================
# CUSTOM PURGED TIME SERIES SPLIT (ANTI-LEAKAGE)
# =============================================================================
class PurgedTimeSeriesSplit:
    """
    Time Series Cross-Validation với gap để tránh data leakage
    
    Parameters:
    -----------
    n_splits : int
        Số lượng folds
    gap : int
        Số rows bỏ qua giữa train và validation (purge window)
        Ví dụ: gap=7 nghĩa là bỏ 7 ngày sau train, trước khi validation bắt đầu
    
    Example:
    --------
    |-------- Train --------|  GAP (7 days)  |---- Val ----|
       Fit pipeline here         (⛔)          Evaluate here
    """
    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Tính kích thước mỗi fold
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Train: từ đầu đến split point
            train_end = (i + 1) * test_size
            
            # Gap: loại bỏ `gap` rows sau train
            val_start = train_end + self.gap
            val_end = val_start + test_size
            
            # Đảm bảo không vượt quá dataset
            if val_end > n_samples:
                break
            
            train_indices = indices[:train_end]
            val_indices = indices[val_start:val_end]
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# =============================================================================
# ALIGNMENT FUNCTION (SỬA LỖI DROPNA)
# =============================================================================
def align_data_for_tuning(X_raw, y_raw, pipeline, scaler, fit_transform=False):
    """
    Standard align function: Run pipeline, scale, and join with y to dropna
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
    
    # ✅✅✅ SỬA LỖI Ở ĐÂY ✅✅✅
    # Chỉ drop các hàng mà CỘT TARGET (y) bị NaN
    # (Chúng ta giả định X_df đã sạch NaNs nhờ pipeline)
    combined_clean = combined.dropna(subset=[y_aligned.name]) 
    
    y_final = combined_clean[y_aligned.name]
    X_final = combined_clean.drop(columns=[y_aligned.name])
    
    return X_final, y_final

# =============================================================================
# DATA LOADING (FIXED - THÊM BƯỚC DROPNA QUAN TRỌNG)
# =============================================================================
def load_data_for_tuning(target_name):
    """
    ✅ FIXED: Gộp, xử lý datetime VÀ dropna TRƯỚC KHI CV.
    """
    print(f"🔍 Loading RAW data (Train + Val COMBINED) for {target_name}...")
    
    # 1. Load CSVs (Không parse, không set index)
    train_df = pd.read_csv(
        os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv")
    )
    val_df = pd.read_csv(
        os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv")
    )
    
    # 2. Gộp lại
    all_train_data = pd.concat([train_df, val_df], ignore_index=True)
    
    # 3. Xử lý Datetime (Logic cũ của bạn, rất tốt)
    if 'datetime' not in all_train_data.columns:
         raise KeyError("❌ Không tìm thấy cột 'datetime' trong file CSV.")
    
    print(f"   ...Sử dụng cột 'datetime' làm cột thời gian")
    all_train_data['datetime'] = pd.to_datetime(all_train_data['datetime'])
    all_train_data = all_train_data.set_index('datetime', drop=False)

    # 4. Sắp xếp lại (Quan trọng cho CV)
    all_train_data = all_train_data.sort_index()

    # 5. ✅✅✅ SỬA LỖI QUAN TRỌNG NHẤT ✅✅✅
    # Xóa bất kỳ hàng nào có giá trị NaN trong cột target
    # (do lỗi từ file Excel gốc)
    rows_before = len(all_train_data)
    all_train_data = all_train_data.dropna(subset=[target_name])
    rows_after = len(all_train_data)
    if rows_before > rows_after:
        print(f"   ⚠️ Đã xóa {rows_before - rows_after} hàng có NaN trong cột target.")

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
    ✅ FIXED: Objective function với Purged TimeSeriesSplit
    
    Đảm bảo:
    - Mỗi fold có pipeline và scaler riêng
    - Gap giữa train và val để tránh leakage
    - Temporal order được bảo toàn
    """
    ranges = config.LINEAR_PARAM_RANGES
    
    # 1. Suggest model và hyperparameters
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

    # ✅ 2. Purged Time Series CV với gap
    tscv = PurgedTimeSeriesSplit(
        n_splits=config.CV_N_SPLITS, 
        gap=config.CV_GAP_DAYS
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
        
        assert train_dates.max() < val_dates.min(), \
            f"❌ FOLD {fold_num+1} LEAKAGE DETECTED! Train ends {train_dates.max()}, Val starts {val_dates.min()}"
        
        # Log first fold for verification
        if fold_num == 0:
            gap_days = (val_dates.min() - train_dates.max()).days
            print(f"  ✅ Fold 1 verified:")
            print(f"     Train: {train_dates.min().date()} → {train_dates.max().date()}")
            print(f"     Gap:   {gap_days} days")
            print(f"     Val:   {val_dates.min().date()} → {val_dates.max().date()}")

        # ✅ 4. Create NEW pipeline and scaler for this fold
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

        # 7. Train model
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
# MAIN OPTUNA SEARCH FUNCTION
# =============================================================================
def run_optuna_search_linear():
    """
    Run Optuna hyperparameter search for all forecast horizons
    """
    # Initialize ClearML task
    task = Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        task_name=config.CLEARML_TASK_NAME,
        tags=["Optuna", "LinearRegression", "Multi-Horizon", "Purged-CV", "Leakage-Free"]
    )
    
    all_best_params = {}
    all_best_scores = {}
    all_best_details = {}

    # Loop through all forecast horizons
    for target_name in config.TARGET_FORECAST_COLS:
        print("\n" + "="*80)
        print(f"🎯 TUNING FOR: {target_name}")
        print("="*80)
    
        # Load raw data
        X_all_train_raw, y_all_train_raw = load_data_for_tuning(target_name)
        
        print(f"🔍 Starting Optuna search...")
        print(f"   Strategy: {config.CV_N_SPLITS}-Fold Purged TimeSeriesSplit (Gap={config.CV_GAP_DAYS} days)")
        print(f"   Trials:   {config.OPTUNA_TRIALS}")
        print(f"   ⚠️  This will take time (re-fits pipelines in each fold)...\n")
        
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
            "gap_days": config.CV_GAP_DAYS
        }

        # Print results
        print(f"\n🏆 BEST RESULTS FOR {target_name}:")
        print(f"   Model Type:    {best_params.get('model_type')}")
        print(f"   Avg Val RMSE:  {val_rmse:.4f} (across {config.CV_N_SPLITS} folds)")
        if 'alpha' in best_params:
            print(f"   Best Alpha:    {best_params.get('alpha'):.6f}")
        if 'l1_ratio' in best_params:
            print(f"   Best L1 Ratio: {best_params.get('l1_ratio'):.4f}")
    
    # Save results to YAML
    output = {
        "best_params": all_best_params,
        "best_scores": all_best_scores,
        "details": all_best_details,
        "config": {
            "n_trials": config.OPTUNA_TRIALS,
            "cv_strategy": "PurgedTimeSeriesSplit",
            "n_splits": config.CV_N_SPLITS,
            "gap_days": config.CV_GAP_DAYS,
            "search_space": config.LINEAR_PARAM_RANGES,
            "leakage_safe": True
        }
    }
    
    output_path = os.path.join(config.MODEL_DIR, config.OPTUNA_RESULTS_YAML)
    with open(output_path, "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ OPTUNA LINEAR SEARCH COMPLETE (LEAKAGE-FREE)")
    print("="*80)
    print(f"📁 Best params saved to: {output_path}")
    print(f"\n📊 Summary of Best RMSE:")
    for target, score in all_best_scores.items():
        print(f"   {target}: {score:.4f}")
    print(f"\n🚀 NEXT STEP: Run 'python train_linear.py' to train final models")
    print("="*80)
    
    task.close()


if __name__ == "__main__":
    run_optuna_search_linear()