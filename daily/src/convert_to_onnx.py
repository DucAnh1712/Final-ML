# convert_to_onnx.py
import os
import joblib
import pandas as pd
import config  # Import your config file
import xgboost # Import xgboost để giúp skl2onnx nhận diện

# Import thư viện ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# CHÚNG TA KHÔNG DÙNG onnxmltools NỮA
# skl2onnx (phiên bản mới) đã có thể tự xử lý XGBoost

def main():
    """
    Chuyển đổi pipeline (Scaler + Model) đã huấn luyện sang định dạng ONNX.
    """
    print("🚀 Starting ONNX conversion...")

    # 1. Tải pipeline có thể chuyển đổi (Scaler + Model)
    model_path = os.path.join(config.MODEL_DIR, "onnx_convertible_pipeline.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please run train.py first to create 'onnx_convertible_pipeline.pkl'")
        return

    model = joblib.load(model_path)
    print(f"✅ Loaded pipeline from: {model_path}")

    # 2. Định nghĩa "hình dạng" (shape) của dữ liệu đầu vào
    feature_test_path = os.path.join(config.FEATURE_DIR, "feature_test.csv")
    if not os.path.exists(feature_test_path):
        print(f"❌ Error: {feature_test_path} not found. Please run feature_engineering.py.")
        return

    df_test = pd.read_csv(feature_test_path)

    if config.TARGET_COL in df_test.columns:
        df_test = df_test.drop(columns=[config.TARGET_COL])

    df_test = df_test.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    num_features = len(df_test.columns)
    print(f"Detected {num_features} input features for the ONNX model.")

    initial_type = [('float_input', FloatTensorType([None, num_features]))]

    # 3. Chuyển đổi model
    print("⚙️ Converting pipeline to ONNX format...")
    try:
        onnx_model = convert_sklearn(
            model,
            "hcm_temperature_model",
            initial_types=initial_type,
            target_opset=12 
        )

        # 4. Lưu model ONNX
        onnx_path = os.path.join(config.MODEL_DIR, "model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"\n🎉 Success! Model saved to: {onnx_path}")

    except Exception as e:
        print(f"❌ Conversion failed. Error: {e}")
        print("If this fails again, please double check your library versions.")

if __name__ == "__main__":
    main()