# optuna_search_lightgbm.py
import os
import pandas as pd
import numpy as np
import yaml
import optuna
import lightgbm as lgb # ⬅️ THAY ĐỔI
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
    # (Copy code của class PurgedTimeSeriesSplit vào đây)
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
# ALIGNMENT FUNCTION (SỬA LỖI DROPNA)
# =============================================================================
def align_data_for_tuning(X_raw, y_raw, pipeline, scaler, fit_transform=False):
    if fit_transform:
        X_feat = pipeline.fit_transform(X_raw)
        X_scaled = scaler.fit_transform(X_feat)
    else:
        X_feat = pipeline.transform(X_raw)
        X_scaled = scaler.transform(X_feat)
    y_aligned = y_raw.copy()
    y_aligned.index = X_feat.index 
    y_df = pd.DataFrame(y_aligned)
    X_df = pd.DataFrame(X_scaled, index=X_feat.index, columns=X_feat.columns) 
    combined = pd.concat([y_df, X_df], axis=1)
    combined_clean = combined.dropna(subset=[y_aligned.name]) 
    y_final = combined_clean[y_aligned.name]
    X_final = combined_clean.drop(columns=[y_aligned.name])
    return X_final, y_final

# =============================================================================
# DATA LOADING (FIXED - THÊM BƯỚC DROPNA QUAN TRỌNG)
# =============================================================================
def load_data_for_tuning(target_name):
    print(f"🔍 Loading RAW data (Train + Val COMBINED) for {target_name}...")
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    all_train_data = pd.concat([train_df, val_df], ignore_index=True)
    if 'datetime' not in all_train_data.columns:
         raise KeyError("❌ Không tìm thấy cột 'datetime' trong file CSV.")
    print(f"   ...Sử dụng cột 'datetime' làm cột thời gian")
    all_train_data['datetime'] = pd.to_datetime(all_train_data['datetime'])
    all_train_data = all_train_data.set_index('datetime', drop=False)
    all_train_data = all_train_data.sort_index()
    rows_before = len(all_train_data)
    all_train_data = all_train_data.dropna(subset=[target_name])
    rows_after = len(all_train_data)
    if rows_before > rows_after:
        print(f"   ⚠️ Đã xóa {rows_before - rows_after} hàng có NaN trong cột target.")
    y_train_full_raw = all_train_data[target_name]
    X_train_full_raw = all_train_data.copy()
    print(f"✅ Combined RAW data shapes: X={X_train_full_raw.shape}, y={y_train_full_raw.shape}")
    print(f"📅 Date range: {X_train_full_raw.index.min()} → {X_train_full_raw.index.max()}")
    return X_train_full_raw, y_train_full_raw

# =============================================================================
# ⬅️ THAY ĐỔI 1: HÀM OBJECTIVE CHO LIGHTGBM
# =============================================================================
def lightgbm_objective(trial, X_all_train_raw, y_all_train_raw):
    """
    Objective function cho LightGBM
    """
    ranges = config.LIGHTGBM_PARAM_RANGES
    
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
        'verbose': -1 # Tắt bớt log
    }

    tscv = PurgedTimeSeriesSplit(
        n_splits=config.CV_N_SPLITS, 
        gap=config.CV_GAP_ROWS
    )
    fold_scores = []
    
    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X_all_train_raw)):
        X_train_fold_raw = X_all_train_raw.iloc[train_idx]
        y_train_fold_raw = y_all_train_raw.iloc[train_idx]
        X_val_fold_raw = X_all_train_raw.iloc[val_idx]
        y_val_fold_raw = y_all_train_raw.iloc[val_idx]
        
        if fold_num == 0:
            train_dates_col = X_train_fold_raw['datetime'] 
            val_dates_col = X_val_fold_raw['datetime']
            gap_duration = (val_dates_col.min() - train_dates_col.max())
            print(f"   ✅ Fold 1 verified:")
            print(f"      Train: {train_dates_col.min()} → {train_dates_col.max()}")
            print(f"      Gap:   {gap_duration} (approx {config.CV_GAP_DAYS} days)")
            print(f"      Val:   {val_dates_col.min()} → {val_dates_col.max()}")

        feature_pipeline_fold = create_feature_pipeline()
        scaler_fold = RobustScaler()
        X_train_fold, y_train_fold = align_data_for_tuning(
            X_train_fold_raw, y_train_fold_raw, 
            feature_pipeline_fold, scaler_fold, 
            fit_transform=True
        )
        X_val_fold, y_val_fold = align_data_for_tuning(
            X_val_fold_raw, y_val_fold_raw, 
            feature_pipeline_fold, scaler_fold, 
            fit_transform=False
        )

        model = lgb.LGBMRegressor(**params) # ⬅️ Đổi model
        
        if X_train_fold.empty or y_train_fold.empty:
            print(f"   ⚠️ Fold {fold_num+1} rỗng. Bỏ qua.")
            continue

        model.fit(X_train_fold, y_train_fold)
        
        y_val_pred = model.predict(X_val_fold)
        val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        fold_scores.append(val_rmse)
    
    final_rmse = np.mean(fold_scores)
    trial.set_user_attr("val_rmse", float(final_rmse))
    
    return final_rmse

# =============================================================================
# ⬅️ THAY ĐỔI 2: HÀM MAIN
# =============================================================================
def run_optuna_search_lightgbm(): # ⬅️ Đổi tên hàm
    task = Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        task_name="Optuna LightGBM (Hourly)",
        tags=["Optuna", "LightGBM", "Multi-Horizon", "Purged-CV", "Hourly"] # ⬅️ Đổi Tag
    )
    
    all_best_params = {}
    all_best_scores = {}
    all_best_details = {}

    for target_name in config.TARGET_FORECAST_COLS:
        print("\n" + "="*80)
        print(f"🎯 TUNING LIGHTGBM FOR: {target_name}") # ⬅️ Đổi tên
        print("="*80)
    
        X_all_train_raw, y_all_train_raw = load_data_for_tuning(target_name)
        
        print(f"🔍 Starting Optuna search (LightGBM)...")
        print(f"   Strategy: {config.CV_N_SPLITS}-Fold Purged TimeSeriesSplit (Gap={config.CV_GAP_ROWS} rows/hours)")
        print(f"   Trials:   {config.OPTUNA_TRIALS}")
        print(f"   ⚠️  This will take time...\n")
        
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: lightgbm_objective(trial, X_all_train_raw, y_all_train_raw), # ⬅️ Gọi hàm objective mới
            n_trials=config.OPTUNA_TRIALS,
            show_progress_bar=True
        )
        
        best_trial = study.best_trial
        best_params = best_trial.params
        val_rmse = best_trial.user_attrs.get("val_rmse", 0)
        all_best_params[target_name] = best_params
        all_best_scores[target_name] = float(val_rmse)
        all_best_details[target_name] = {"val_rmse": float(val_rmse), "n_folds": config.CV_N_SPLITS, "gap_rows": config.CV_GAP_ROWS}

        print(f"\n🏆 BEST LIGHTGBM RESULTS FOR {target_name}:") # ⬅️ Đổi tên
        print(f"   Avg Val RMSE: {val_rmse:.4f} (across {config.CV_N_SPLITS} folds)")
        print(f"   Best n_estimators: {best_params.get('n_estimators')}")
        print(f"   Best learning_rate: {best_params.get('learning_rate'):.6f}")

    output = {
        "best_params": all_best_params,
        "best_scores": all_best_scores,
        "details": all_best_details,
        "config": {
            "n_trials": config.OPTUNA_TRIALS,
            "cv_strategy": "PurgedTimeSeriesSplit",
            "n_splits": config.CV_N_SPLITS,
            "gap_rows": config.CV_GAP_ROWS,
            "search_space": config.LIGHTGBM_PARAM_RANGES, # ⬅️ Đổi tên
            "leakage_safe": True
        }
    }
    
    # ⬅️ THAY ĐỔI 3: Tên file output
    output_path = os.path.join(config.MODEL_DIR, config.OPTUNA_RESULTS_LIGHTGBM_YAML)
    with open(output_path, "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)
    
    print("\n" + "="*80)
    print("✅ OPTUNA LIGHTGBM SEARCH COMPLETE (LEAKAGE-FREE)")
    print("="*80)
    print(f"📁 Best params saved to: {output_path}")
    print(f"\n📊 Summary of Best RMSE:")
    for target, score in all_best_scores.items():
        print(f"   {target}: {score:.4f}")
    
    # ⬅️ THAY ĐỔI 4: Bước tiếp theo
    print(f"\n🚀 NEXT STEP: Run 'python train_lightgbm.py' to train final models")
    print("="*80)
    
    task.close()

if __name__ == "__main__":
    run_optuna_search_lightgbm() # ⬅️ Đổi tên hàm