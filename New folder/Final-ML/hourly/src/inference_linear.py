import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import config

def load_production_models():
    # (Hàm này đã chuẩn, không cần sửa)
    print("Loading production components...")
    
    pipeline_path = os.path.join(config.MODEL_DIR, config.PIPELINE_NAME)
    scaler_path = os.path.join(config.MODEL_DIR, config.SCALER_NAME)
    
    if not all(os.path.exists(p) for p in [pipeline_path, scaler_path]):
        raise FileNotFoundError("feature_pipeline.pkl or scaler.pkl not found. Please run train_linear.py first.")
        
    pipeline = joblib.load(pipeline_path)
    scaler = joblib.load(scaler_path)
    
    models = {}
    for target_name in config.TARGET_FORECAST_COLS:
        model_name = f"{target_name}_{config.MODEL_NAME_LINEAR}"
        model_path = os.path.join(config.MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found. Please run train_linear.py.")
        models[target_name] = joblib.load(model_path)
        
    print(f"✅ Pipeline, Scaler, and {len(models)} models loaded.")
    return pipeline, scaler, models

def load_test_data():
    """
    ✅ SỬA LỖI: Load test data (với DatetimeIndex)
    Giống hệt logic của 'optuna_search_linear.py'
    """
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv")
    
    # ✅ SỬA LỖI: Load file CSV với DatetimeIndex
    df_test = pd.read_csv(
        test_path,
        index_col=0,      # Dùng cột 0 (chính là 'datetime') làm index
        parse_dates=[0]   # Ép kiểu cột 0 thành datetime
    )
    
    # ✅ SỬA LỖI: Sắp xếp và đảm bảo cột 'datetime' tồn tại
    df_test = df_test.sort_index()
    if 'datetime' not in df_test.columns:
         df_test['datetime'] = df_test.index
    
    print(f"✅ Test data loaded: {df_test.shape}")
    
    X_test_raw = df_test.copy()
    # df_test contains the target columns (y_actual)
    return X_test_raw, df_test 

def calculate_metrics(y_actual, y_pred):
    # (Hàm này đã chuẩn, không cần sửa)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_actual, y_pred))),
        "MAE": float(mean_absolute_error(y_actual, y_pred)),
        "R2": float(r2_score(y_actual, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_actual, y_pred))
    }

def main():
    pipeline, scaler, models = load_production_models()
    X_test_raw, df_test_full = load_test_data()

    # ✅ SỬA LỖI: Đảm bảo X_test_raw có DatetimeIndex
    # (Đã được sửa trong hàm load_test_data())
    print(f"\n--- Evaluating {config.TARGET_FORECAST_COLS} ---")

    # 1. Run pipeline (Transform, Scale) ONCE
    print(f"⚙️ Transforming test features...")
    # X_test_raw giờ đã có DatetimeIndex chuẩn
    X_feat_test = pipeline.transform(X_test_raw)
    print(f"⚙️ Scaling test features...")
    X_scaled_test = scaler.transform(X_feat_test)
    
    X_scaled_test_df = pd.DataFrame(X_scaled_test, index=X_feat_test.index, columns=X_feat_test.columns)

    all_metrics = {}
    all_predictions = {}
    
    # ✅ SỬA LỖI: Đảm bảo chúng ta lưu DatetimeIndex
    all_predictions['datetime'] = X_scaled_test_df.index 

    # ======================================================
    # 2. LOOP & PREDICT FOR EACH HORIZON (Đã chuẩn, không cần sửa)
    # ======================================================
    for target_name in config.TARGET_FORECAST_COLS: # Tự động lặp qua t24, t48,...
        print(f"\n--- Predicting {target_name} ---")
        model = models[target_name]
        
        # 3. Align y_actual (Get from raw data)
        # df_test_full giờ đã có DatetimeIndex chuẩn
        y_actual_raw = df_test_full[target_name]
        y_actual_aligned = y_actual_raw.copy()
        
        # ✅ SỬA LỖI: Căn chỉnh y_actual_aligned với index của X
        # (Vì pipeline có thể drop vài hàng đầu tiên do lag/rolling)
        y_actual_aligned = y_actual_aligned.loc[X_feat_test.index]

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

    # ✅ SỬA LỖI: Đọc tên file output từ config.py
    metrics_path = os.path.join(config.OUTPUT_DIR, config.TEST_METRICS_LINEAR_NAME)
    with open(metrics_path, "w") as f:
        yaml.dump(all_metrics, f, sort_keys=False)
    print(f"\n💾 All test metrics saved to: {metrics_path}")
    
    # ✅ SỬA LỖI: Đọc tên file output từ config.py
    pred_path = os.path.join(config.OUTPUT_DIR, config.TEST_PREDS_LINEAR_NAME)
    df_preds = pd.DataFrame(all_predictions)
    
    # ✅ SỬA LỖI: Set 'datetime' làm index khi lưu file
    df_preds = df_preds.set_index('datetime')
    df_preds.to_csv(pred_path)
    print(f"💾 All predictions saved to: {pred_path}")
    
    print("\n🎉 Multi-Horizon Linear evaluation complete.")

if __name__ == "__main__":
    main()