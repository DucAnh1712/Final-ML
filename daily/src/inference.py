# inference_linear.py (MULTI-HORIZON VERSION)
import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import config

def load_production_models():
    """Load pipeline, scaler, and ALL horizon models"""
    print("Loading production components...")
    
    pipeline_path = os.path.join(config.MODEL_DIR, config.PIPELINE_NAME)
    scaler_path = os.path.join(config.MODEL_DIR, config.SCALER_NAME)
    
    # Check the 2 common files
    if not all(os.path.exists(p) for p in [pipeline_path, scaler_path]):
        raise FileNotFoundError("feature_pipeline.pkl or scaler.pkl not found. Please run train_linear.py first.")
        
    pipeline = joblib.load(pipeline_path)
    scaler = joblib.load(scaler_path)
    
    # ✅ CHANGE: Load 7 models
    models = {}
    for target_name in config.TARGET_FORECAST_COLS: # Loop through 7 targets
        model_name = f"{target_name}_{config.MODEL_NAME}"
        model_path = os.path.join(config.MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found. Please run train_linear.py.")
        models[target_name] = joblib.load(model_path)
        
    print(f"✅ Pipeline, Scaler, and {len(models)} models loaded.")
    return pipeline, scaler, models

def load_test_data():
    """Load test data (processed)"""
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv")
    df_test = pd.read_csv(test_path)
    df_test['datetime'] = pd.to_datetime(df_test['datetime'])
    
    print(f"✅ Test data loaded: {df_test.shape}")
    
    X_test_raw = df_test.copy()
    # df_test contains the target columns (y_actual)
    return X_test_raw, df_test 

def calculate_metrics(y_actual, y_pred):
    """Calculate metrics (with float() cast)."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_actual, y_pred))),
        "MAE": float(mean_absolute_error(y_actual, y_pred)),
        "R2": float(r2_score(y_actual, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_actual, y_pred))
    }

def main():
    pipeline, scaler, models = load_production_models()
    X_test_raw, df_test_full = load_test_data()

    print(f"\n--- Evaluating {config.TARGET_FORECAST_COLS} ---")

    # 1. Run pipeline (Transform, Scale) ONCE
    print(f"⚙️ Transforming test features...")
    X_feat_test = pipeline.transform(X_test_raw)
    print(f"⚙️ Scaling test features...")
    X_scaled_test = scaler.transform(X_feat_test)
    
    # Wrap X_scaled into DataFrame (with index)
    X_scaled_test_df = pd.DataFrame(X_scaled_test, index=X_feat_test.index, columns=X_feat_test.columns)

    all_metrics = {}
    all_predictions = {}
    all_predictions['datetime'] = X_scaled_test_df.index # Save index

    # ======================================================
    # 2. LOOP & PREDICT FOR EACH HORIZON (✅ CHANGE)
    # ======================================================
    for target_name in config.TARGET_FORECAST_COLS: # Loop 7 times
        print(f"\n--- Predicting {target_name} ---")
        model = models[target_name] # Get the corresponding model
        
        # 3. Align y_actual (Get from raw data)
        y_actual_raw = df_test_full[target_name]
        y_actual_aligned = y_actual_raw.copy()
        y_actual_aligned.index = X_feat_test.index # Assign DatetimeIndex

        # 4. Predict
        y_pred_raw = model.predict(X_scaled_test_df)
        
        # 5. Align y_pred and dropna
        pred_col_name = f"pred_{target_name}"
        y_pred_series = pd.Series(y_pred_raw, index=y_actual_aligned.index, name=pred_col_name)
        
        combined = pd.concat([y_actual_aligned, y_pred_series], axis=1)
        combined_clean = combined.dropna() # Drop NaNs (from target shift)
        
        y_actual_clean = combined_clean[target_name]
        y_pred_clean = combined_clean[pred_col_name]
        
        # 6. Calculate Metrics
        metrics = calculate_metrics(y_actual_clean, y_pred_clean)
        all_metrics[target_name] = metrics
        
        print(f"📊 Test Set Performance ({target_name}):")
        for k, v in metrics.items():
            print(f"   {k:<6}: {v:.4f}")
            
        # 7. Save predictions (for review)
        all_predictions[target_name] = y_actual_aligned
        all_predictions[pred_col_name] = y_pred_series

    # Save all test metrics
    metrics_path = os.path.join(config.OUTPUT_DIR, "test_metrics_linear.yaml")
    with open(metrics_path, "w") as f:
        yaml.dump(all_metrics, f, sort_keys=False)
    print(f"\n💾 All test metrics saved to: {metrics_path}")
    
    # Save all predictions
    pred_path = os.path.join(config.OUTPUT_DIR, "test_predictions_linear.csv")
    df_preds = pd.DataFrame(all_predictions)
    df_preds.to_csv(pred_path)
    print(f"💾 All predictions saved to: {pred_path}")
    
    print("\n🎉 Multi-Horizon Linear evaluation complete.")

if __name__ == "__main__":
    main()