# inference_xgboost.py
import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import config
# (Không cần import xgb, chỉ load file pkl)

def load_production_models():
    print("Loading production components (XGBoost)...") # ⬅️ Sửa log
    
    pipeline_path = os.path.join(config.MODEL_DIR, config.PIPELINE_NAME)
    scaler_path = os.path.join(config.MODEL_DIR, config.SCALER_NAME)
    
    if not all(os.path.exists(p) for p in [pipeline_path, scaler_path]):
        raise FileNotFoundError("pipeline/scaler not found. Please run train_xgboost.py first.")
        
    pipeline = joblib.load(pipeline_path)
    scaler = joblib.load(scaler_path)
    
    models = {}
    for target_name in config.TARGET_FORECAST_COLS:
        # ✅ SỬA 1: Dùng tên model XGBOOST
        model_name = f"{target_name}_{config.MODEL_NAME_XGBOOST}"
        model_path = os.path.join(config.MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found. Please run train_xgboost.py.")
        models[target_name] = joblib.load(model_path)
        
    print(f"✅ Pipeline, Scaler, and {len(models)} XGBoost models loaded.") # ⬅️ Sửa log
    return pipeline, scaler, models

# (Hàm load_test_data và calculate_metrics giữ nguyên y hệt)
def load_test_data():
    # (Copy y hệt từ file linear)
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv")
    df_test = pd.read_csv(test_path, index_col=0, parse_dates=[0])
    df_test = df_test.sort_index()
    if 'datetime' not in df_test.columns:
         df_test['datetime'] = df_test.index
    print(f"✅ Test data loaded: {df_test.shape}")
    X_test_raw = df_test.copy()
    return X_test_raw, df_test 

def calculate_metrics(y_actual, y_pred):
    # (Copy y hệt từ file linear)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_actual, y_pred))),
        "MAE": float(mean_absolute_error(y_actual, y_pred)),
        "R2": float(r2_score(y_actual, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_actual, y_pred))
    }

def main():
    # (Tất cả logic trong main() giữ nguyên y hệt,
    # chỉ sửa 2 dòng lưu file cuối cùng)
    
    pipeline, scaler, models = load_production_models()
    X_test_raw, df_test_full = load_test_data()
    print(f"\n--- Evaluating {config.TARGET_FORECAST_COLS} (XGBoost) ---") # ⬅️ Sửa log
    
    # 1. Run pipeline
    print(f"⚙️ Transforming test features...")
    X_feat_test = pipeline.transform(X_test_raw)
    print(f"⚙️ Scaling test features...")
    X_scaled_test = scaler.transform(X_feat_test)
    X_scaled_test_df = pd.DataFrame(X_scaled_test, index=X_feat_test.index, columns=X_feat_test.columns)
    all_metrics = {}
    all_predictions = {}
    all_predictions['datetime'] = X_scaled_test_df.index 

    # 2. LOOP & PREDICT
    for target_name in config.TARGET_FORECAST_COLS:
        print(f"\n--- Predicting {target_name} ---")
        model = models[target_name]
        y_actual_raw = df_test_full[target_name]
        y_actual_aligned = y_actual_raw.copy()
        y_actual_aligned = y_actual_aligned.loc[X_feat_test.index]
        y_pred_raw = model.predict(X_scaled_test_df)
        pred_col_name = f"pred_{target_name}"
        y_pred_series = pd.Series(y_pred_raw, index=y_actual_aligned.index, name=pred_col_name)
        combined = pd.concat([y_actual_aligned, y_pred_series], axis=1)
        combined_clean = combined.dropna()
        y_actual_clean = combined_clean[target_name]
        y_pred_clean = combined_clean[pred_col_name]
        metrics = calculate_metrics(y_actual_clean, y_pred_clean)
        all_metrics[target_name] = metrics
        print(f"📊 Test Set Performance ({target_name}):")
        for k, v in metrics.items():
            print(f"   {k:<6}: {v:.4f}")
        all_predictions[target_name] = y_actual_aligned
        all_predictions[pred_col_name] = y_pred_series

    # ✅ SỬA 2: Dùng tên file metrics XGBOOST
    metrics_path = os.path.join(config.OUTPUT_DIR, config.TEST_METRICS_XGBOOST_NAME)
    with open(metrics_path, "w") as f:
        yaml.dump(all_metrics, f, sort_keys=False)
    print(f"\n💾 All test metrics saved to: {metrics_path}")
    
    # ✅ SỬA 3: Dùng tên file predictions XGBOOST
    pred_path = os.path.join(config.OUTPUT_DIR, config.TEST_PREDS_XGBOOST_NAME)
    df_preds = pd.DataFrame(all_predictions)
    df_preds = df_preds.set_index('datetime')
    df_preds.to_csv(pred_path)
    print(f"💾 All predictions saved to: {pred_path}")
    
    print("\n🎉 Multi-Horizon XGBoost evaluation complete.") # ⬅️ Sửa log

if __name__ == "__main__":
    main()