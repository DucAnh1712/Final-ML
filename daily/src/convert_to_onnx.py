# convert_to_onnx.py
import os
import joblib
import pandas as pd
import config
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def get_num_features_from_scaler(scaler_path):
    """
    Loads the saved scaler and returns the number of features it expects.
    """
    if not os.path.exists(scaler_path):
        print(f"❌ Error: Scaler not found at {scaler_path}")
        print(f"Please run train_linear.py first to create '{config.SCALER_NAME}'")
        return None
    
    try:
        scaler = joblib.load(scaler_path)
        num_features = scaler.n_features_in_
        print(f"✅ Loaded scaler. Detected {num_features} input features.")
        return num_features
    except Exception as e:
        print(f"❌ Error loading scaler or getting n_features_in_: {e}")
        return None

def main():
    """
    Converts all 7 regression models (T+1 to T+7) to the ONNX format.
    """
    print("🚀 Starting ONNX conversion for multiple Models...")

    # 1. Load the scaler to get the feature count
    scaler_path = os.path.join(config.MODEL_DIR, config.SCALER_NAME)
    num_features = get_num_features_from_scaler(scaler_path)
    
    if num_features is None:
        return

    # 2. Define the input "shape" for the models
    # This is the input *AFTER* preprocessing (after pipeline and scaler)
    # [None, num_features] means: (arbitrary batch_size, fixed number of features)
    initial_type = [('float_input', FloatTensorType([None, num_features]))]

    # 3. Iterate and convert each model
    for target_name in config.TARGET_FORECAST_COLS: # Loop for each target (e.g., T+1 to T+7)
        print("\n" + "="*50)
        print(f"🎯 Converting model for: {target_name}")
        
        # 3a. Load the .pkl model
        model_pkl_name = f"{target_name}_{config.MODEL_NAME}"
        model_pkl_path = os.path.join(config.MODEL_DIR, model_pkl_name)
        
        if not os.path.exists(model_pkl_path):
            print(f"❌ Error: Model not found at {model_pkl_path}")
            print(f"Skipping {target_name}...")
            continue
            
        model = joblib.load(model_pkl_path)
        print(f"✅ Loaded model: {model_pkl_name}")

        # 3b. Convert the model to ONNX
        onnx_model_name = f"{target_name}_{config.MODEL_NAME}.onnx"
        onnx_model_path = os.path.join(config.MODEL_DIR, onnx_model_name)
        
        print(f"⚙️ Converting {model.__class__.__name__} to ONNX format...")
        try:
            onnx_model = convert_sklearn(
                model,
                f"model_{target_name}",
                initial_types=initial_type,
                target_opset=12 
            )

            # 3c. Save the ONNX model
            with open(onnx_model_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"🎉 Success! Model saved to: {onnx_model_path}")

        except Exception as e:
            print(f"❌ Conversion failed for {target_name}. Error: {e}")

    print("\n" + "="*50)
    print("✅ ONNX conversion process complete for all models.")

if __name__ == "__main__":
    main()