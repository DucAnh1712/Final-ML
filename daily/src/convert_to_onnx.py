# convert_to_onnx.py
import os
import joblib
import pandas as pd
import config  # Import your config file

# Import thư viện ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def get_num_features_from_scaler(scaler_path):
    """
    Tải scaler đã lưu và trả về số lượng features nó mong đợi.
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
    Chuyển đổi tất cả 7 model hồi quy (T+1 đến T+7) sang định dạng ONNX.
    """
    print("🚀 Starting ONNX conversion for 7 Linear Models...")

    # 1. Tải scaler để lấy số lượng features
    scaler_path = os.path.join(config.MODEL_DIR, config.SCALER_NAME)
    num_features = get_num_features_from_scaler(scaler_path)
    
    if num_features is None:
        return

    # 2. Định nghĩa "hình dạng" (shape) của dữ liệu đầu vào cho các model
    # Đây là đầu vào *SAU KHI* đã qua tiền xử lý (đã qua pipeline và scaler)
    # [None, num_features] có nghĩa là: (batch_size tùy ý, số lượng features cố định)
    initial_type = [('float_input', FloatTensorType([None, num_features]))]

    # 3. Lặp và chuyển đổi từng model
    for target_name in config.TARGET_FORECAST_COLS: # Lặp 7 lần
        print("\n" + "="*50)
        print(f"🎯 Converting model for: {target_name}")
        
        # 3a. Tải model .pkl
        model_pkl_name = f"{target_name}_{config.MODEL_NAME}"
        model_pkl_path = os.path.join(config.MODEL_DIR, model_pkl_name)
        
        if not os.path.exists(model_pkl_path):
            print(f"❌ Error: Model not found at {model_pkl_path}")
            print(f"Skipping {target_name}...")
            continue
            
        model = joblib.load(model_pkl_path)
        print(f"✅ Loaded model: {model_pkl_name}")

        # 3b. Chuyển đổi model sang ONNX
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

            # 3c. Lưu model ONNX
            with open(onnx_model_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"🎉 Success! Model saved to: {onnx_model_path}")

        except Exception as e:
            print(f"❌ Conversion failed for {target_name}. Error: {e}")

    print("\n" + "="*50)
    print("✅ ONNX conversion process complete for all models.")

if __name__ == "__main__":
    main()