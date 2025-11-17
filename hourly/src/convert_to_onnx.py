# hourly/src/convert_to_onnx.py
import os
import joblib
import pandas as pd
import config
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler # Explicitly import for clarity

# Import ONNX libraries
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def main():
    """
    Loads the Scaler and Model separately,
    combines them into a Pipeline, then converts the Pipeline to ONNX.
    """
    print("🚀 Starting ONNX conversion...")

    # 1. Load the two components
    scaler_path = os.path.join(config.MODEL_DIR, "scaler_for_onnx.pkl")
    model_json_path = os.path.join(config.MODEL_DIR, "model_for_onnx.json")

    if not os.path.exists(scaler_path) or not os.path.exists(model_json_path):
        print(f"❌ Error: Missing model files.")
        print("Please run train.py first to create 'scaler_for_onnx.pkl' and 'model_for_onnx.json'")
        return

    # 1a. Load Scaler
    scaler = joblib.load(scaler_path)
    print(f"✅ Loaded Scaler from: {scaler_path}")
    
    # 1b. Load Model (using load_model function)
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model(model_json_path)
    print(f"✅ Loaded XGBoost Model from: {model_json_path}")

    # 2. Combine the two components into an in-memory Pipeline
    model_pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model_xgb)
    ])

    # 3. Define the input shape
    # We use a test feature file to infer the number of features.
    # Note: Assuming feature_test.csv is stored in PROCESSED_DATA_DIR
    feature_test_path = os.path.join(config.PROCESSED_DATA_DIR, "feature_test.csv")
    
    if not os.path.exists(feature_test_path):
        print(f"❌ Error: Missing test feature file at {feature_test_path}. Cannot infer input shape.")
        return

    df_test = pd.read_csv(feature_test_path)
    
    # Keep only numeric columns (safest approach for ONNX conversion)
    df_test = df_test.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    num_features = len(df_test.columns)
    
    print(f"Detected {num_features} input features for the ONNX model.")
    # Define the input type: FloatTensorType([batch_size, num_features])
    initial_type = [('float_input', FloatTensorType([None, num_features]))]

    # 4. Convert the combined pipeline
    print("⚙️ Converting pipeline to ONNX format...")
    try:
        onnx_model = convert_sklearn(
            model_pipeline,
            "hcm_temperature_model",
            initial_types=initial_type,
            target_opset=12 # Common ONNX standard version
        )

        # 5. Save the ONNX model
        onnx_path = os.path.join(config.MODEL_DIR, "model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"\n🎉 Success! Model saved to: {onnx_path}")

    except Exception as e:
        print(f"❌ Conversion failed. Error: {e}")

if __name__ == "__main__":
    main()