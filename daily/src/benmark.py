import os
import time
import joblib
import numpy as np
import onnxruntime as ort
import config  # Tệp config.py của bạn

# --- THÊM VÀO ---
# Import các thư viện để lưu kết quả
import pandas as pd
import yaml
import warnings

# Bỏ qua các cảnh báo không cần thiết
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# --- KẾT THÚC THÊM VÀO ---


# ======================================================
# 1. CÀI ĐẶT
# ======================================================
MODEL_DIR = config.MODEL_DIR

PIPELINE_FILENAME = 'onnx_convertible_pipeline.pkl'
PIPELINE_PATH = os.path.join(MODEL_DIR, PIPELINE_FILENAME)

# Chọn một mô hình để benchmark
TARGET_DAY = config.TARGET_FORECAST_COLS[0] # Ví dụ: 'target_T+1'
MODEL_NAME = f"{TARGET_DAY}_{config.MODEL_NAME}" # Ví dụ: 'target_T+1_model_daily'

PKL_PATH = os.path.join(MODEL_DIR, MODEL_NAME) # <-- Đã thêm .pkl
ONNX_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx")

# Cấu hình benchmark
N_SAMPLES = 1000  # Số lượng mẫu trong 1 lô
N_ITERATIONS = 100 # Chạy bao nhiêu lần để lấy trung bình


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
        full_preprocessing_pipeline = joblib.load(pipeline_path)
        feature_pipeline = full_preprocessing_pipeline.named_steps['feature_engineering']
        preprocessor_step = feature_pipeline.named_steps['preprocess_columns']
        num_features = len(preprocessor_step.final_cols)
        
        print(f"✅ Loaded pipeline. Detected {num_features} output features from ColumnPreprocessor.")
        return num_features
        
    except Exception as e:
        print(f"❌ Error loading pipeline or getting feature count: {e}")
        return None

# ======================================================
# 2. TẠO DỮ LIỆU ĐẦU VÀO GIẢ
# ======================================================
print("🚀 Starting Inference Benchmark...")
num_features = get_num_features_from_pipeline(PIPELINE_PATH)
if num_features is None:
    exit()

print(f"Creating dummy data: ({N_SAMPLES}, {num_features}) features.")
dummy_data = np.random.rand(N_SAMPLES, num_features).astype(np.float32)

# ======================================================
# 3. TẢI CÁC MODEL
# ======================================================

print("Loading models...")
# 3a. Tải model Sklearn (.pkl)
try:
    model_sklearn = joblib.load(PKL_PATH)
    print(f"✅ Successfully loaded {PKL_PATH}")
except Exception as e:
    print(f"❌ Error loading {PKL_PATH}: {e}")
    exit()

# 3b. Tải model ONNX (cho CPU)
try:
    sess_onnx_cpu = ort.InferenceSession(
        ONNX_PATH, 
        providers=['CPUExecutionProvider']
    )
    input_name = sess_onnx_cpu.get_inputs()[0].name
    print(f"✅ Successfully loaded {ONNX_PATH} for CPU.")
except Exception as e:
    print(f"❌ Error loading {ONNX_PATH} for CPU: {e}")
    exit()

# 3c. Tải model ONNX (cho GPU)
sess_onnx_gpu = None
try:
    sess_onnx_gpu = ort.InferenceSession(
        ONNX_PATH, 
        providers=['CUDAExecutionProvider']
    )
    print(f"✅ Successfully loaded {ONNX_PATH} for GPU (CUDA).")
except Exception as e:
    print(f"⚠️ Could not load model for GPU (CUDA): {e}")
    print("   Make sure you have 'onnxruntime-gpu' installed and have NVIDIA/CUDA drivers.")

# ======================================================
# 4. CHẠY BENCHMARK
# ======================================================
print("\n" + "="*50)
print(f"Running benchmark with {N_SAMPLES} samples, repeating {N_ITERATIONS} times...")
print("="*50)

# 4a. Benchmark Sklearn (CPU)
start_time = time.perf_counter()
for _ in range(N_ITERATIONS):
    _ = model_sklearn.predict(dummy_data)
end_time = time.perf_counter()
sklearn_time = (end_time - start_time) / N_ITERATIONS
print(f"⏱️ Sklearn (CPU) : {sklearn_time * 1000:.6f} ms / batch")

# 4b. Benchmark ONNX (CPU)
start_time = time.perf_counter()
for _ in range(N_ITERATIONS):
    _ = sess_onnx_cpu.run(None, {input_name: dummy_data})
end_time = time.perf_counter()
onnx_cpu_time = (end_time - start_time) / N_ITERATIONS
print(f"⏱️ ONNX (CPU)    : {onnx_cpu_time * 1000:.6f} ms / batch")

# 4c. Benchmark ONNX (GPU)
onnx_gpu_time = None # Khởi tạo
if sess_onnx_gpu:
    start_time = time.perf_counter()
    for _ in range(N_ITERATIONS):
        _ = sess_onnx_gpu.run(None, {input_name: dummy_data})
    end_time = time.perf_counter()
    onnx_gpu_time = (end_time - start_time) / N_ITERATIONS
    print(f"⏱️ ONNX (GPU)    : {onnx_gpu_time * 1000:.6f} ms / batch")

# ======================================================
# 5. HIỂN THỊ VÀ LƯU KẾT QUẢ (PHONG CÁCH MỚI)
# ======================================================
print("\n" + "="*70)
print("🏆 FINAL INFERENCE BENCHMARK RESULTS 🏆")
print("="*70)

# 1. Xây dựng danh sách kết quả
results = []

results.append({
    "Method": "Sklearn (CPU)",
    "Time_ms_per_batch": sklearn_time * 1000,
    "Speedup_vs_Sklearn": 1.0  # Baseline
})

results.append({
    "Method": "ONNX (CPU)",
    "Time_ms_per_batch": onnx_cpu_time * 1000,
    "Speedup_vs_Sklearn": sklearn_time / onnx_cpu_time
})

if onnx_gpu_time:
    results.append({
        "Method": "ONNX (GPU)",
        "Time_ms_per_batch": onnx_gpu_time * 1000,
        "Speedup_vs_Sklearn": sklearn_time / onnx_gpu_time
    })

# 2. Chuyển sang DataFrame để in
results_df = pd.DataFrame(results).sort_values(by="Time_ms_per_batch")
print(results_df.to_string(index=False, float_format="%.4f"))

# 3. Lưu file YAML
# BẠN CẦN THÊM BIẾN NÀY VÀO config.py
INFERENCE_BENCHMARK_YAML = "inference_benchmark.yaml" 
# Hoặc, nếu bạn đã định nghĩa nó trong config, hãy dùng:
# INFERENCE_BENCHMARK_YAML = config.INFERENCE_BENCHMARK_YAML

output_path = os.path.join(config.OUTPUT_DIR, INFERENCE_BENCHMARK_YAML)

# Chuyển đổi sang dict
results_dict = results_df.to_dict('records')

try:
    with open(output_path, "w") as f:
        yaml.dump(results_dict, f, sort_keys=False)
    print(f"\n💾 Inference benchmark results saved to: {output_path}")
except Exception as e:
    print(f"\n❌ Error saving inference benchmark results: {e}")