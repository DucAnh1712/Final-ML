# hourly/src/benmark.py
import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
from lightgbm import LGBMRegressor
import xgboost as xgb
import config
from feature_engineering import create_feature_pipeline 
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- HELPER FUNCTIONS ---

def set_datetime_idx(df, file_name):
    """Converts the 'datetime' column to a DatetimeIndex and sorts the DataFrame."""
    if 'datetime' not in df.columns:
        raise KeyError(f"❌ Error: 'datetime' column not found in file {file_name}")
    
    # Ensure all data in the column is treated as datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime', drop=False)
    df = df.sort_index()
    return df

# --- MAIN BENCHMARK LOGIC ---

def run_benchmark():
    print(f"🚀 STARTING MODEL BENCHMARK (V3 - Ridge, DecisionTree, Ensembles)")
    print("="*70)
    print(f"Target: {config.TARGET_FORECAST_COLS[0]}")
    print("Features: Derived Features (Pipeline from feature_engineering.py)")
    print("="*70)

    # 1. LOAD DATA (Train/Val/Test)
    print("📂 Loading all data...")
    train_df_raw = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df_raw = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    test_df_raw = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv"))
    
    target_name = config.TARGET_FORECAST_COLS[0]

    # Set Datetime Index for all dataframes
    train_df_raw = set_datetime_idx(train_df_raw, "data_train.csv")
    val_df_raw = set_datetime_idx(val_df_raw, "data_val.csv")
    test_df_raw = set_datetime_idx(test_df_raw, "data_test.csv")
    
    # Clean up NaN in target column for safe splitting
    train_df = train_df_raw.dropna(subset=[target_name]).copy()
    val_df = val_df_raw.dropna(subset=[target_name]).copy()
    test_df = test_df_raw.dropna(subset=[target_name]).copy()
    print(f"   ...Cleaned NaN in target column.")

    # Assign X and y (now with DatetimeIndex)
    y_train_raw = train_df[target_name] 
    X_train_raw = train_df.copy()
    
    y_val_raw = val_df[target_name]
    X_val_raw = val_df.copy()

    y_test_raw = test_df[target_name]
    X_test_raw = test_df.copy()

    # 2. PREPARE PIPELINE & DATA (Fit/Transform)
    print("🛠️ Preparing data (Fitting NEW pipeline on Train)...")
    feature_pipeline_fit = create_feature_pipeline()
    scaler_fit = RobustScaler()

    # --- TRAIN ---
    # 1. Transform X_train_raw -> X_train_feat (Feature Engineering)
    X_train_feat = feature_pipeline_fit.fit_transform(X_train_raw) 
    
    # 2. Scale X_train_feat -> X_train_final (Scaling)
    X_train_final = pd.DataFrame(
        scaler_fit.fit_transform(X_train_feat), 
        index=X_train_feat.index, 
        columns=X_train_feat.columns
    )
    
    # 3. Align y_train_final using the clean index from X_train_final
    y_train_final = y_train_raw.loc[X_train_final.index] 

    # --- VAL ---
    X_val_feat = feature_pipeline_fit.transform(X_val_raw)
    X_val_final = pd.DataFrame(
        scaler_fit.transform(X_val_feat), 
        index=X_val_feat.index, 
        columns=X_val_feat.columns
    )
    y_val_final = y_val_raw.loc[X_val_final.index]
    
    # --- TEST ---
    X_test_feat = feature_pipeline_fit.transform(X_test_raw)
    X_test_final = pd.DataFrame(
        scaler_fit.transform(X_test_feat), 
        index=X_test_feat.index, 
        columns=X_test_feat.columns
    )
    y_test_final = y_test_raw.loc[X_test_final.index]
    
    print(f"📊 Train data: X={X_train_final.shape}, y={y_train_final.shape}")
    print(f"📊 Val data: X={X_val_final.shape}, y={y_val_final.shape}")
    print(f"📊 Test data: X={X_test_final.shape}, y={y_test_final.shape}")
    
    # 3. DEFINE MODELS
    models = {
        "LinearRegression": LinearRegression(n_jobs=-1),
        
        "Ridge": Ridge(random_state=42), 
        
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10, 
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        ),
        
        "LightGBM": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=42
        ),
        
        "XGBoost": xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=42,
            # Note: early_stopping_rounds should be set inside the fit method for XGBoost 
            # when using eval_set, but the class parameter is used as a fallback/default.
        )
    }

    results = []

    # 4. RUN BENCHMARK
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Training logic adapted for Early Stopping on Tree Boosters
        if name == "LightGBM":
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        elif name == "XGBoost":
            # Using eval_set with the defined early_stopping_rounds in the constructor
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                early_stopping_rounds=100, 
                verbose=False
            )
        else:
            model.fit(X_train_final, y_train_final)
        
        # Predict on all sets
        y_train_pred = model.predict(X_train_final)
        y_val_pred = model.predict(X_val_final)
        y_test_pred = model.predict(X_test_final)
        
        # Calculate Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_final, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val_final, y_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_final, y_test_pred))
        
        print(f"✅ {name} Done. Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        
        results.append({
            "Model": name,
            "Train_RMSE": train_rmse,
            "Val_RMSE": val_rmse,
            "Test_RMSE": test_rmse,
            # Calculate Overfitting Gap relative to Train RMSE
            "Gap_Val (%)": (val_rmse - train_rmse) / train_rmse * 100,
            "Gap_Test (%)": (test_rmse - train_rmse) / train_rmse * 100
        })

    # 5. SHOW & SAVE RESULTS
    print("\n" + "="*70)
    print("🏆 FINAL BENCHMARK RESULTS (DERIVED FEATURES V3) 🏆")
    print("="*70)
    
    results_df = pd.DataFrame(results).sort_values(by="Test_RMSE")
    print(results_df.to_string(index=False, float_format="%.4f"))

    # Save YAML file
    output_path = os.path.join(config.OUTPUT_DIR, config.BENCHMARK_RESULTS_YAML)
    try:
        # Convert to a list of dicts for clean YAML saving
        results_dict = results_df.to_dict('records')
        with open(output_path, "w") as f:
            yaml.dump(results_dict, f, sort_keys=False)
        print(f"\n💾 Benchmark results saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Error saving benchmark results: {e}")

if __name__ == "__main__":
    run_benchmark()