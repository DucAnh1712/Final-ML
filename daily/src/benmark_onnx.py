import os
import time
import joblib
import numpy as np
import onnxruntime as ort
import config  # Tệp config.py của bạn

# ======================================================
# 1. CÀI ĐẶT
# ======================================================
MODEL_DIR = config.MODEL_DIR

# THAY ĐỔI: Tên tệp pipeline mà train.py đã tạo
PIPELINE_FILENAME = 'onnx_convertible_pipeline.pkl'
PIPELINE_PATH = os.path.join(MODEL_DIR, PIPELINE_FILENAME)

# Chọn một mô hình để benchmark
TARGET_DAY = config.TARGET_FORECAST_COLS[0] # Ví dụ: 'target_T+1'
MODEL_NAME = f"{TARGET_DAY}_{config.MODEL_NAME}" # Ví dụ: 'target_T+1_model_daily'

PKL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}")
ONNX_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx")

# Cấu hình benchmark
N_SAMPLES = 1000  # Số lượng mẫu trong 1 lô
N_ITERATIONS = 100 # Chạy bao nhiêu lần để lấy trung bình


# THAY ĐỔI: Hàm này giống hệt hàm trong 'convert_to_onnx.py'
def get_num_features_from_pipeline(pipeline_path):
    """
    Tải pipeline tiền xử lý và đếm số lượng features đầu ra
    từ bước 'preprocess_columns'.
    """
    if not os.path.exists(pipeline_path):
        print(f"❌ Error: Pipeline not found at {pipeline_path}")
        print(f"Please run train_linear.py first to create '{PIPELINE_FILENAME}'")
        return None
    
    try:
        # Tải pipeline đầy đủ ( [feature_engineering_pipeline], [scaler] )
        full_preprocessing_pipeline = joblib.load(pipeline_path)
        
        # Truy cập vào pipeline feature engineering bên trong
        feature_pipeline = full_preprocessing_pipeline.named_steps['feature_engineering']
        
        # Truy cập vào bước cuối cùng (ColumnPreprocessor)
        preprocessor_step = feature_pipeline.named_steps['preprocess_columns']
        
        # Lấy danh sách cột cuối cùng đã được 'fit'
        num_features = len(preprocessor_step.final_cols)
        
        print(f"✅ Loaded pipeline. Detected {num_features} output features from ColumnPreprocessor.")
        return num_features
        
    except Exception as e:
        print(f"❌ Error loading pipeline or getting feature count: {e}")
        return None

# ======================================================
# 2. TẠO DỮ LIỆU ĐẦU VÀO GIẢ
# ======================================================
print("🚀 Starting Benchmark...")
# THAY ĐỔI: Sử dụng hàm mới
num_features = get_num_features_from_pipeline(PIPELINE_PATH)
if num_features is None:
    exit()

print(f"Tạo dữ liệu giả: ({N_SAMPLES}, {num_features}) features.")
# Dữ liệu này giả lập là ĐÃ QUA pipeline và scaler
dummy_data = np.random.rand(N_SAMPLES, num_features).astype(np.float32)

# ======================================================
# 3. TẢI CÁC MODEL
# ======================================================

print("Loading models...")
# 3a. Tải model Sklearn (.pkl)
try:
    model_sklearn = joblib.load(PKL_PATH)
    print(f"✅ Tải thành công {PKL_PATH}")
except Exception as e:
    print(f"❌ Lỗi tải {PKL_PATH}: {e}")
    exit()

# 3b. Tải model ONNX (cho CPU)
try:
    sess_onnx_cpu = ort.InferenceSession(
        ONNX_PATH, 
        providers=['CPUExecutionProvider'] # Chỉ định rõ chạy trên CPU
    )
    input_name = sess_onnx_cpu.get_inputs()[0].name
    print(f"✅ Tải thành công {ONNX_PATH} cho CPU.")
except Exception as e:
    print(f"❌ Lỗi tải {ONNX_PATH} cho CPU: {e}")
    exit()

# 3c. Tải model ONNX (cho GPU)
sess_onnx_gpu = None
try:
    sess_onnx_gpu = ort.InferenceSession(
        ONNX_PATH, 
        providers=['CUDAExecutionProvider'] # Chỉ định rõ chạy trên GPU
    )
    print(f"✅ Tải thành công {ONNX_PATH} cho GPU (CUDA).")
except Exception as e:
    print(f"⚠️ Không thể tải model cho GPU (CUDA): {e}")
    print("   Hãy đảm bảo bạn đã cài 'onnxruntime-gpu' và có driver NVIDIA/CUDA.")

# ======================================================
# 4. CHẠY BENCHMARK
# ======================================================
print("\n" + "="*50)
print(f"Chạy benchmark với {N_SAMPLES} mẫu, lặp lại {N_ITERATIONS} lần...")
print("="*50)

# 4a. Benchmark Sklearn (CPU)
start_time = time.perf_counter()
for _ in range(N_ITERATIONS):
    _ = model_sklearn.predict(dummy_data)
end_time = time.perf_counter()
sklearn_time = (end_time - start_time) / N_ITERATIONS
print(f"⏱️ Sklearn (CPU) : {sklearn_time * 1000:.6f} ms / lô")

# 4b. Benchmark ONNX (CPU)
start_time = time.perf_counter()
for _ in range(N_ITERATIONS):
    _ = sess_onnx_cpu.run(None, {input_name: dummy_data})
end_time = time.perf_counter()
onnx_cpu_time = (end_time - start_time) / N_ITERATIONS
print(f"⏱️ ONNX (CPU)    : {onnx_cpu_time * 1000:.6f} ms / lô")

# 4c. Benchmark ONNX (GPU)
if sess_onnx_gpu:
    start_time = time.perf_counter()
    for _ in range(N_ITERATIONS):
        _ = sess_onnx_gpu.run(None, {input_name: dummy_data})
    end_time = time.perf_counter()
    onnx_gpu_time = (end_time - start_time) / N_ITERATIONS
    print(f"⏱️ ONNX (GPU)    : {onnx_gpu_time * 1000:.6f} ms / lô")

# ======================================================
# 5. KẾT LUẬN
# ======================================================
print("\n" + "="*50)
print("Kết luận:")
factor_cpu = sklearn_time / onnx_cpu_time
print(f"🎉 ONNX (CPU) nhanh hơn Sklearn (CPU) **{factor_cpu:.2f} lần**.")

if sess_onnx_gpu:
    factor_gpu = sklearn_time / onnx_gpu_time
    print(f"🎉 ONNX (GPU) nhanh hơn Sklearn (CPU) **{factor_gpu:.2f} lần**.")