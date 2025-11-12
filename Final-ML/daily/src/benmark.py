# benchmark.py (V3 - Adding Ridge & DecisionTree)
import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ ADD IMPORTS
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor # ✅ New model
from lightgbm import LGBMRegressor
import xgboost as xgb
import config
from feature_engineering import create_feature_pipeline # ✅ Import NEW pipeline
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def align_data_v2(X_raw, y_raw, pipeline, scaler, fit_transform=False):
    """New align function, compatible with NEW pipeline (FIXED)"""
    
    # 1. Run pipeline (Input is X_raw, output is X_feat)
    if fit_transform:
        X_feat = pipeline.fit_transform(X_raw)
        X_scaled = scaler.fit_transform(X_feat)
    else:
        X_feat = pipeline.transform(X_raw)
        X_scaled = scaler.transform(X_feat)
    
    # 2. Align y
    y_aligned = y_raw.copy()
    y_aligned.index = X_feat.index # Assign DatetimeIndex to y
    
    # 3. Repack to dropna
    y_df = pd.DataFrame(y_aligned)
    X_df = pd.DataFrame(X_scaled, index=X_feat.index, columns=X_feat.columns) 
    
    combined = pd.concat([y_df, X_df], axis=1)
    combined_clean = combined.dropna()
    
    y_clean = combined_clean[y_aligned.name]
    X_clean = combined_clean.drop(columns=[y_aligned.name])
    
    return X_clean, y_clean

def run_benchmark():
    print(f"🚀 STARTING MODEL BENCHMARK (V3 - Adding Ridge/Tree)")
    print("="*70)
    print(f"Features: Derived Features (from Colab)")
    print("="*70)

    # ======================================================
    # 1. LOAD DATA (Train/Val/Test)
    # ======================================================
    print("📂 Loading all data...")
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    test_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv"))
    
    target_name = config.TARGET_FORECAST_COLS[0]
    
    y_train_raw = train_df[target_name]
    X_train_raw = train_df.copy()
    
    y_val_raw = val_df[target_name]
    X_val_raw = val_df.copy()

    y_test_raw = test_df[target_name]
    X_test_raw = test_df.copy()

    # ======================================================
    # 2. PREPARE PIPELINE & DATA (Fit/Transform)
    # ======================================================
    print("🛠️ Preparing data (Fitting NEW pipeline on Train)...")
    feature_pipeline_fit = create_feature_pipeline() # NEW Pipeline
    scaler_fit = RobustScaler()

    # Fit_transform on Train
    X_train_final, y_train_final = align_data_v2(
        X_train_raw, y_train_raw, 
        feature_pipeline_fit, scaler_fit, fit_transform=True
    )
    
    # Transform on Val
    X_val_final, y_val_final = align_data_v2(
        X_val_raw, y_val_raw, 
        feature_pipeline_fit, scaler_fit, fit_transform=False
    )
    
    # Transform on Test
    X_test_final, y_test_final = align_data_v2(
        X_test_raw, y_test_raw, 
        feature_pipeline_fit, scaler_fit, fit_transform=False
    )
    
    print(f"📊 Train data: X={X_train_final.shape}, y={y_train_final.shape}")
    print(f"📊 Val data: X={X_val_final.shape}, y={y_val_final.shape}")
    print(f"📊 Test data: X={X_test_final.shape}, y={y_test_final.shape}")

    # ======================================================
    # 3. DEFINE MODELS
    # ======================================================
    models = {
        "LinearRegression": LinearRegression(n_jobs=-1),
        
        "Ridge": Ridge(random_state=42), # ✅ New model
        
        "DecisionTree": DecisionTreeRegressor(random_state=42), # ✅ New model
        
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
            early_stopping_rounds=100
        )
    }

    results = []

    # ======================================================
    # 4. RUN BENCHMARK
    # ======================================================
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        if name == "LightGBM":
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        elif name == "XGBoost":
             model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                verbose=False
            )
        else:
            model.fit(X_train_final, y_train_final)
        
        # Predict
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
            "Gap_Val (%)": (val_rmse - train_rmse) / train_rmse * 100,
            "Gap_Test (%)": (test_rmse - train_rmse) / train_rmse * 100
        })

    # ======================================================
    # 5. SHOW & SAVE RESULTS
    # ======================================================
    print("\n" + "="*70)
    print("🏆 FINAL BENCHMARK RESULTS (DERIVED FEATURES V3) 🏆")
    print("="*70)
    
    results_df = pd.DataFrame(results).sort_values(by="Test_RMSE")
    print(results_df.to_string(index=False, float_format="%.4f"))

    # 1. Convert to dict to save YAML
    results_dict = results_df.to_dict('records')
    
    # 2. Save YAML file
    output_path = os.path.join(config.OUTPUT_DIR, config.BENCHMARK_RESULTS_YAML)
    try:
        with open(output_path, "w") as f:
            yaml.dump(results_dict, f, sort_keys=False)
        print(f"\n💾 Benchmark results saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Error saving benchmark results: {e}")

if __name__ == "__main__":
    run_benchmark()