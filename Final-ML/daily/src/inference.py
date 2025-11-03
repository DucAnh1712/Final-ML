# inference.py
import os
import pandas as pd
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import config

def load_production_model():
    """Tải pipeline sản phẩm (features + scaler + model)."""
    model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}. Please run train.py first.")
    
    model = joblib.load(model_path)
    print(f"✅ Production pipeline loaded from: {model_path}")
    return model

def load_test_data():
    """Tải dữ liệu test (processed, trước khi tạo feature)."""
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ data_test.csv not found at {test_path}. Please run data_processing.py first.")
    
    df = pd.read_csv(test_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    print(f"✅ Test data loaded: {df.shape}")
    
    # Tách X_test và y_test
    # y_test chính là cột 'temp'
    # X_test là tất cả các cột còn lại (dữ liệu thô)
    X_test = df.drop(columns=[config.TARGET_COL], errors='ignore')
    y_test = df[config.TARGET_COL]
    
    return X_test, y_test, df

def evaluate_on_test(model, X_test, y_test):
    """
    Đánh giá model trên tập test (1-step-ahead forecast).
    Step 5: Dùng các metrics RMSE, MAPE, R2
    """
    print("⚙️ Predicting on test set...")
    # Pipeline sẽ tự động chạy:
    # 1. feature_engineering (FFillImputer -> TimeFeatures -> LagRolling -> DropRaw)
    # 2. scaler (RobustScaler)
    # 3. model (XGBRegressor)
    y_pred = model.predict(X_test)

    # Xử lý các giá trị NaN có thể được tạo ra ở đầu (do rolling)
    # Chúng ta cần căn chỉnh y_test và y_pred
    
    # Tìm số hàng NaN ở đầu y_pred (nếu có)
    # (Pipeline của chúng ta đã xử lý .dropna() bên trong)
    # Nhưng X_test gốc có thể dài hơn y_pred
    
    if len(y_pred) < len(y_test):
        print(f"Aligning predictions: Dropping first {len(y_test) - len(y_pred)} rows from y_test to match rolling window NaNs.")
        # Bỏ đi các hàng đầu của y_test, tương ứng với các hàng NaN đã bị drop
        y_test = y_test.iloc[len(y_test) - len(y_pred):]
    

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred)
    }

    print("\n📊 Test Set Performance (1-step-ahead):")
    for k, v in metrics.items():
        print(f"   {k:<6}: {v:.4f}")
        
    return y_pred, y_test, metrics # Trả về y_test đã được căn chỉnh

def save_results(df_test, y_test_aligned, y_pred, metrics, output_dir):
    """Lưu kết quả dự đoán và metrics."""
    
    # Chỉ lấy các hàng của df_test tương ứng với y_test đã căn chỉnh
    result_df = df_test.iloc[len(df_test) - len(y_test_aligned):].copy()
    
    result_df["predicted_temp"] = y_pred

    pred_path = os.path.join(output_dir, "test_predictions.csv")
    metrics_path = os.path.join(output_dir, "test_metrics.yaml")

    result_df.to_csv(pred_path, index=False)
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, sort_keys=False)

    print(f"\n💾 Predictions saved to: {pred_path}")
    print(f"🧾 Metrics saved to: {metrics_path}")
    return result_df

def visualize_predictions(df_results, output_dir):
    """Vẽ biểu đồ so sánh thực tế vs dự đoán."""
    
    plt.figure(figsize=(15, 6))
    plt.plot(df_results['datetime'], df_results[config.TARGET_COL], label="Actual", marker='.', linestyle='-')
    plt.plot(df_results['datetime'], df_results['predicted_temp'], label="Predicted", marker='x', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Test Set Performance (Actual vs. Predicted)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    line_path = os.path.join(output_dir, "test_predictions_line_plot.png")
    plt.savefig(line_path)
    print(f"📈 Line plot saved to: {line_path}")

def main():
    model = load_production_model()
    X_test, y_test, df_test = load_test_data()
    y_pred, y_test_aligned, metrics = evaluate_on_test(model, X_test, y_test)
    df_results = save_results(df_test, y_test_aligned, y_pred, metrics, config.OUTPUT_DIR)
    visualize_predictions(df_results, config.OUTPUT_DIR)
    print("\n🎉 Test set evaluation complete.")

if __name__ == "__main__":
    main()