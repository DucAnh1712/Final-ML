# benmark_onnx.py
import os
import time
import joblib
import numpy as np
import onnxruntime as ort
import config

MODEL_DIR = config.MODEL_DIR

PIPELINE_FILENAME = 'onnx_convertible_pipeline.pkl'
PIPELINE_PATH = os.path.join(MODEL_DIR, PIPELINE_FILENAME)

# Assuming the first target is used for this specific model benchmark (e.g., target_t1)
TARGET_DAY = config.TARGET_FORECAST_COLS[0]
MODEL_NAME = f"{TARGET_DAY}_{config.MODEL_NAME}"

PKL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}")
ONNX_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx")

N_SAMPLES = 1000  # Number of samples (rows) in the dummy data batch
N_ITERATIONS = 100  # Number of times to run the prediction for averaging


def get_num_features_from_pipeline(pipeline_path):
    """
    Loads the preprocessing pipeline and counts the number of output features
    from the 'preprocess_columns' step.
    """
    if not os.path.exists(pipeline_path):
        print(f"❌ Error: Pipeline not found at {pipeline_path}")
        print(f"Please run train_linear.py first to create '{PIPELINE_FILENAME}'")
        return None
    
    try:
        # Load the complete pipeline ( [feature_engineering_pipeline], [scaler] )
        full_preprocessing_pipeline = joblib.load(pipeline_path)
        
        # Access the inner feature engineering pipeline
        feature_pipeline = full_preprocessing_pipeline.named_steps['feature_engineering']
        
        # Access the final step (ColumnPreprocessor)
        preprocessor_step = feature_pipeline.named_steps['preprocess_columns']
        
        # Get the length of the final 'fit' columns list
        num_features = len(preprocessor_step.final_cols)
        
        print(f"✅ Loaded pipeline. Detected {num_features} output features from ColumnPreprocessor.")
        return num_features
        
    except Exception as e:
        print(f"❌ Error loading pipeline or getting feature count: {e}")
        return None

# --- SETUP ---
print("🚀 Starting Benchmark...")
num_features = get_num_features_from_pipeline(PIPELINE_PATH)
if num_features is None:
    exit()

print(f"Creating dummy data: ({N_SAMPLES}, {num_features}) features.")
dummy_data = np.random.rand(N_SAMPLES, num_features).astype(np.float32)

print("Loading models...")
try:
    # 1. Load Sklearn Model
    model_sklearn = joblib.load(PKL_PATH)
    print(f"✅ Successfully loaded {PKL_PATH}")
except Exception as e:
    print(f"❌ Error loading {PKL_PATH}: {e}")
    exit()

try:
    # 2. Load ONNX Model (CPU)
    sess_onnx_cpu = ort.InferenceSession(
        ONNX_PATH, 
        providers=['CPUExecutionProvider'] # Explicitly specify CPU
    )
    input_name = sess_onnx_cpu.get_inputs()[0].name
    print(f"✅ Successfully loaded {ONNX_PATH} for CPU.")
except Exception as e:
    print(f"❌ Error loading {ONNX_PATH} for CPU: {e}")
    exit()

# 3. Load ONNX Model (GPU - Optional)
sess_onnx_gpu = None
try:
    sess_onnx_gpu = ort.InferenceSession(
        ONNX_PATH, 
        providers=['CUDAExecutionProvider'] # Explicitly specify GPU
    )
    print(f"✅ Successfully loaded {ONNX_PATH} for GPU (CUDA).")
except Exception as e:
    print(f"⚠️ Could not load model for GPU (CUDA): {e}")
    print("   Ensure you have installed 'onnxruntime-gpu' and have NVIDIA/CUDA drivers.")

# --- BENCHMARK EXECUTION ---
print("\n" + "="*50)
print(f"Running benchmark with {N_SAMPLES} samples, repeated {N_ITERATIONS} times...")
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
    # ONNX inference requires the input data be passed as a dictionary {input_name: data}
    _ = sess_onnx_cpu.run(None, {input_name: dummy_data})
end_time = time.perf_counter()
onnx_cpu_time = (end_time - start_time) / N_ITERATIONS
print(f"⏱️ ONNX (CPU)    : {onnx_cpu_time * 1000:.6f} ms / batch")

# 4c. Benchmark ONNX (GPU)
if sess_onnx_gpu:
    start_time = time.perf_counter()
    for _ in range(N_ITERATIONS):
        # ONNX inference
        _ = sess_onnx_gpu.run(None, {input_name: dummy_data})
    end_time = time.perf_counter()
    onnx_gpu_time = (end_time - start_time) / N_ITERATIONS
    print(f"⏱️ ONNX (GPU)    : {onnx_gpu_time * 1000:.6f} ms / batch")

# --- CONCLUSION ---
print("\n" + "="*50)
print("Conclusion:")
factor_cpu = sklearn_time / onnx_cpu_time
print(f"🎉 ONNX (CPU) is **{factor_cpu:.2f} times** faster than Sklearn (CPU).")

if sess_onnx_gpu:
    factor_gpu = sklearn_time / onnx_gpu_time
    print(f"🎉 ONNX (GPU) is **{factor_gpu:.2f} times** faster than Sklearn (CPU).")