# convert_to_onnx.py
import os
import joblib
import pandas as pd
import config
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Import thư viện ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def main():
    """
    Tải Scaler và Model riêng biệt,
    ghép chúng lại thành Pipeline, rồi chuyển đổi sang ONNX.
    """
    print("🚀 Starting ONNX conversion...")

    # 1. Tải 2 file components
    scaler_path = os.path.join(config.MODEL_DIR, "scaler_for_onnx.pkl")
    model_json_path = os.path.join(config.MODEL_DIR, "model_for_onnx.json")

    if not os.path.exists(scaler_path) or not os.path.exists(model_json_path):
        print(f"❌ Error: Missing model files.")
        print("Please run train.py first to create 'scaler_for_onnx.pkl' and 'model_for_onnx.json'")
        return

    # 1a. Tải Scaler
    scaler = joblib.load(scaler_path)
    print(f"✅ Loaded Scaler from: {scaler_path}")
    
    # 1b. Tải Model (dùng hàm load_model)
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model(model_json_path)
    print(f"✅ Loaded XGBoost Model from: {model_json_path}")

    # 2. "Ghép" 2 file lại thành một Pipeline trong bộ nhớ
    model_pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model_xgb)
    ])

    # 3. Định nghĩa "hình dạng" (shape) đầu vào
    feature_test_path = os.path.join(config.FEATURE_DIR, "feature_test.csv")
    df_test = pd.read_csv(feature_test_path)
    
    # Chỉ giữ lại các cột số (an toàn nhất)
    df_test = df_test.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    num_features = len(df_test.columns)
    
    print(f"Detected {num_features} input features for the ONNX model.")
    initial_type = [('float_input', FloatTensorType([None, num_features]))]

    # 4. Chuyển đổi pipeline đã "ghép"
    print("⚙️ Converting pipeline to ONNX format...")
    try:
        onnx_model = convert_sklearn(
            model_pipeline,
            "hcm_temperature_model",
            initial_types=initial_type,
            target_opset=12
        )

        # 5. Lưu model ONNX
        onnx_path = os.path.join(config.MODEL_DIR, "model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"\n🎉 Success! Model saved to: {onnx_path}")

    except Exception as e:
        print(f"❌ Conversion failed. Error: {e}")

if __name__ == "__main__":
    main()