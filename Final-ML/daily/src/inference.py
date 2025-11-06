# inference.py
import os
import pandas as pd
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import config

def load_production_models():
    """Tải tất cả các pipeline sản phẩm (T+1, T+3, T+5, T+7)."""
    models = {}
    print("Loading production models...")
    for target_name in config.TARGET_FORECAST_COLS:
        model_name = f"{target_name}_pipeline.pkl"
        model_path = os.path.join(config.MODEL_DIR, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}. Please run train.py first.")
        
        models[target_name] = joblib.load(model_path)
        print(f"✅ Loaded pipeline for {target_name}")
    return models

def load_test_data():
    """Tải dữ liệu test (processed), tách X và Y (dict)."""
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ data_test.csv not found at {test_path}. Please run data_processing.py first.")
    
    df_test = pd.read_csv(test_path)
    df_test["datetime"] = pd.to_datetime(df_test["datetime"])
    df_test = df_test.sort_values("datetime").reset_index(drop=True)
    
    print(f"✅ Test data loaded: {df_test.shape}")
    
    # y_test_dict chứa các "đáp án"
    y_test_dict = {}
    for target_name in config.TARGET_FORECAST_COLS:
        y_test_dict[target_name] = df_test[target_name]
        
    # X_test là features, phải drop TẤT CẢ các cột target
    # (Giống hệt logic trong train.py)
    cols_to_drop_prod = config.TARGET_FORECAST_COLS + [config.TARGET_COL]
    X_test = df_test.drop(columns=cols_to_drop_prod, errors='ignore')
    
    # Trả về df_test gốc để dùng cho việc ghép nối (join)
    return X_test, y_test_dict, df_test

def calculate_metrics(y_actual, y_pred):
    """Tính toán bộ metrics."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_actual, y_pred)),
        "MAE": mean_absolute_error(y_actual, y_pred),
        "R2": r2_score(y_actual, y_pred),
        "MAPE": mean_absolute_percentage_error(y_actual, y_pred)
    }

# (Dán đè lên hàm visualize_predictions cũ trong file inference.py)

def visualize_predictions(df_results, output_dir):
    """
    Vẽ biểu đồ so sánh thực tế vs dự đoán (phiên bản "thông minh").
    Nó sẽ kiểm tra xem cột có tồn tại không trước khi vẽ.
    """
    
    plt.figure(figsize=(15, 7))
    
    # 1. Luôn vẽ đường Thực tế (temp gốc)
    if config.TARGET_COL in df_results.columns:
        plt.plot(df_results['datetime'], df_results[config.TARGET_COL], 
                 label="Actual (temp)", color='black', linewidth=2)
    
    # 2. Kiểm tra và vẽ T+1 (Nếu có)
    col_t1 = 'pred_target_T1'
    if col_t1 in df_results.columns:
        plt.plot(df_results['datetime'], df_results[col_t1], 
                 label="Predicted (T+1)", linestyle='--', marker='o', markersize=2)
    
    # 3. Kiểm tra và vẽ T+3 (Nếu có)
    col_t3 = 'pred_target_T3'
    if col_t3 in df_results.columns:
        plt.plot(df_results['datetime'], df_results[col_t3], 
                 label="Predicted (T+3)", linestyle=':', marker='x', markersize=2)
                 
    # 4. Kiểm tra và vẽ T+5 (Nếu có)
    col_t5 = 'pred_target_T5'
    if col_t5 in df_results.columns:
        plt.plot(df_results['datetime'], df_results[col_t5], 
                 label="Predicted (T+5)", linestyle='-.', marker='s', markersize=2)

    # 5. Kiểm tra và vẽ T+7 (Nếu có)
    col_t7 = 'pred_target_T7'
    if col_t7 in df_results.columns:
        plt.plot(df_results['datetime'], df_results[col_t7], 
                 label="Predicted (T+7)", linestyle='--', marker='^', markersize=2)

    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Test Set Performance (Multi-Target Forecast)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    line_path = os.path.join(output_dir, "test_predictions_multitarget_plot.png")
    plt.savefig(line_path)
    print(f"📈 Multi-target line plot saved to: {line_path}")
    
def main():
    models = load_production_models()
    X_test, y_test_dict, df_test_full = load_test_data()

    all_metrics = {}
    # Tạo DataFrame kết quả, bắt đầu với datetime và 'temp' gốc
    df_results = df_test_full[['datetime', config.TARGET_COL]].copy()

    for target_name in config.TARGET_FORECAST_COLS:
        print(f"\n--- Evaluating {target_name} ---")
        model = models[target_name]
        y_actual_raw = y_test_dict[target_name] # (Dài 591, có NaN ở cuối)

        # 1. Dự đoán
        print(f"⚙️ Predicting {target_name}...")
        y_pred_raw = model.predict(X_test) # (Dài 561, do dropna ở đầu)

        # 2. Căn chỉnh (Align)
        
        # 2a. Căn chỉnh START (do rolling dropna)
        # Bỏ đi các hàng đầu của y_actual_raw
        rows_dropped_at_start = len(y_actual_raw) - len(y_pred_raw)
        y_actual_aligned_start = y_actual_raw.iloc[rows_dropped_at_start:]
        
        # 2b. Căn chỉnh END (do shift(-n) dropna)
        # Gói vào DataFrame để dropna cả hai cùng lúc
        df_align = pd.DataFrame({
            'pred': y_pred_raw,
            'actual': y_actual_aligned_start.values
        }, index=y_actual_aligned_start.index)
        
        df_clean = df_align.dropna()
        
        y_pred_clean = df_clean['pred']
        y_actual_clean = df_clean['actual']
        
        print(f"Alignment: Start rows dropped={rows_dropped_at_start}. End rows dropped={len(df_align) - len(df_clean)}.")

        # 3. Tính Metrics
        metrics = calculate_metrics(y_actual_clean, y_pred_clean)
        all_metrics[target_name] = metrics
        
        print(f"📊 Test Set Performance ({target_name}):")
        for k, v in metrics.items():
            print(f"   {k:<6}: {v:.4f}")
            
        # 4. Lưu dự đoán (căn chỉnh) vào df_results
        # Tạo một Series dự đoán (căn chỉnh) với index gốc để join
        pred_series = pd.Series(y_pred_raw, index=y_actual_aligned_start.index)
        df_results[f'pred_{target_name}'] = pred_series

    # 5. Lưu tất cả kết quả
    pred_path = os.path.join(config.OUTPUT_DIR, "test_predictions.csv")
    metrics_path = os.path.join(config.OUTPUT_DIR, "test_metrics.yaml")

    df_results.to_csv(pred_path, index=False)
    with open(metrics_path, "w") as f:
        yaml.dump(all_metrics, f, sort_keys=False)

    print(f"\n💾 All predictions saved to: {pred_path}")
    print(f"🧾 All metrics saved to: {metrics_path}")

    # 6. Vẽ biểu đồ
    visualize_predictions(df_results, config.OUTPUT_DIR)
    print("\n🎉 Multi-target evaluation complete.")

if __name__ == "__main__":
    main()