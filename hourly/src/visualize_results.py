import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import config
import sys

# (Hàm load_one_metric_file và load_all_metrics giữ nguyên như cũ)

def load_one_metric_file(filepath, model_type, metric_type):
    """
    Hàm helper: Tải 1 file YAML, chuyển thành DF, và thêm cột
    """
    if not os.path.exists(filepath):
        print(f"❌ CẢNH BÁO: Không tìm thấy file metrics: {filepath}")
        print("   Hãy chạy file train/inference tương ứng trước.")
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        df = pd.DataFrame.from_dict(data, orient='index')
        df['model_type'] = model_type
        df['metric_type'] = metric_type
        
        # Chuyển 'target_t24', 'target_t48' -> 1, 2, 3... (Ngày)
        horizon_hours = df.index.str.replace('target_t', '').astype(int)
        df['Horizon'] = horizon_hours / 24
        
        return df
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {filepath}: {e}")
        return None

def load_all_metrics():
    """
    Tải TẤT CẢ 6 file metrics (Train/Test của 3 mô hình)
    """
    all_dfs = []
    
    # === 1. Tải mô hình Linear ===
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TRAIN_METRICS_LINEAR_NAME),
        'Linear', 'Train'
    ))
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TEST_METRICS_LINEAR_NAME),
        'Linear', 'Test'
    ))
    
    # === 2. Tải mô hình XGBoost ===
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TRAIN_METRICS_XGBOOST_NAME),
        'XGBoost', 'Train'
    ))
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TEST_METRICS_XGBOOST_NAME),
        'XGBoost', 'Test'
    ))
    
    # === 3. Tải mô hình LightGBM ===
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TRAIN_METRICS_LIGHTGBM_NAME),
        'LightGBM', 'Train'
    ))
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TEST_METRICS_LIGHTGBM_NAME),
        'LightGBM', 'Test'
    ))
    
    # Kiểm tra nếu có file nào bị thiếu
    if any(df is None for df in all_dfs):
        print("\nMột hoặc nhiều file metrics bị thiếu. Dừng chương trình.")
        return None
        
    # Gộp tất cả lại
    full_df = pd.concat(all_dfs)
    
    print("✅ Đã tải và gộp thành công 6 file metrics.")
    return full_df

# ===================================================================
# CÁC HÀM VẼ BIỂU ĐỒ (ĐÃ CẬP NHẬT)
# ===================================================================

def plot_test_metric_comparison(df_test_only, metric_name, title, ylabel, filename, higher_is_better=False):
    """
    Hàm chung để vẽ 3 mô hình cho 1 chỉ số Test (RMSE, MAE, R2)
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Sắp xếp legend theo hiệu suất
    sorted_models = df_test_only.groupby('model_type')[metric_name].mean().sort_values(ascending=higher_is_better).index
    
    sns.lineplot(
        data=df_test_only,
        x='Horizon',
        y=metric_name,
        hue='model_type', # 3 màu cho 3 mô hình
        hue_order=sorted_models, # Sắp xếp legend
        style='model_type', # 3 kiểu đường cho 3 mô hình
        style_order=sorted_models,
        markers=True,
        linewidth=2.5,
        markersize=8
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Ngày dự báo (T+N)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(range(1, 8)) # 1, 2, ... 7
    plt.legend(title="Mô hình (Tốt nhất -> Kém nhất)")
    plt.grid(True, alpha=0.7)

    plot_path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(plot_path, dpi=120)
    print(f"💾 Đã lưu biểu đồ: {plot_path}")

def plot_overfitting_comparison(df):
    """
    Biểu đồ 4: So sánh độ Overfitting (Gap) của 3 mô hình
    """
    # 1. Pivot data để có Train/Test trên cùng 1 hàng
    df_pivot = df.pivot_table(
        index=['Horizon', 'model_type'], 
        columns='metric_type', 
        values='RMSE' # Vẫn dùng RMSE làm gốc
    ).reset_index()
    
    # 2. Tính toán Gap
    df_pivot['Gap (%)'] = (df_pivot['Test'] - df_pivot['Train']) / df_pivot['Train'] * 100
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 3. Vẽ biểu đồ Gap
    sns.lineplot(
        data=df_pivot,
        x='Horizon',
        y='Gap (%)',
        hue='model_type',
        style='model_type',
        markers=True,
        linewidth=2.5,
        markersize=8
    )
    
    plt.title('So sánh Overfitting (Train-Test RMSE Gap) 3 Mô hình (Hourly)', fontsize=16, fontweight='bold')
    plt.xlabel('Ngày dự báo (T+N)', fontsize=12)
    plt.ylabel('Overfitting (Gap %)', fontsize=12)
    plt.xticks(range(1, 8))
    plt.legend(title="Mô hình")
    plt.axhline(0, color='black', linestyle='--', linewidth=1) # Đường 0%
    plt.grid(True, alpha=0.7)

    plot_path = os.path.join(config.OUTPUT_DIR, 'compare_ALL_MODELS_Overfitting_Gap.png')
    plt.savefig(plot_path, dpi=120)
    print(f"💾 Đã lưu biểu đồ Overfitting Gap: {plot_path}")

def main():
    # Load tất cả 6 file
    df_full_metrics = load_all_metrics()
    
    if df_full_metrics is not None:
        # Lọc ra data Test để tái sử dụng
        df_test_only = df_full_metrics[df_full_metrics['metric_type'] == 'Test'].copy()

        # === VẼ 4 BIỂU ĐỒ ===
        
        # 1. Biểu đồ RMSE (Lỗi tuyệt đối)
        plot_test_metric_comparison(
            df_test_only,
            metric_name='RMSE',
            title='So sánh Hiệu suất (Test RMSE) 3 Mô hình (Hourly)',
            ylabel='RMSE (Lỗi nhiệt độ °C)',
            filename='compare_ALL_MODELS_Test_RMSE.png',
            higher_is_better=False # RMSE: Càng thấp càng tốt
        )
        
        # 2. Biểu đồ R2 (Độ "fit")
        plot_test_metric_comparison(
            df_test_only,
            metric_name='R2',
            title='So sánh Độ "Fit" (Test R2) 3 Mô hình (Hourly)',
            ylabel='R-Squared (R²)',
            filename='compare_ALL_MODELS_Test_R2.png',
            higher_is_better=True # R2: Càng cao càng tốt
        )
        
        # 3. Biểu đồ MAE (Lỗi tuyệt đối)
        plot_test_metric_comparison(
            df_test_only,
            metric_name='MAE',
            title='So sánh Hiệu suất (Test MAE) 3 Mô hình (Hourly)',
            ylabel='MAE (Lỗi nhiệt độ °C)',
            filename='compare_ALL_MODELS_Test_MAE.png',
            higher_is_better=False # MAE: Càng thấp càng tốt
        )
        
        # 4. Biểu đồ Overfitting (Dùng df_full_metrics)
        plot_overfitting_comparison(df_full_metrics)
        
        print("\n🎉 HOÀN TẤT TRỰC QUAN HÓA SO SÁNH 3 MÔ HÌNH!")
        print(f"Xem 4 file .png trong: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    # Dependency checks
    try:
        import yaml
    except ImportError:
        print("\nERROR: Missing 'PyYAML' library.")
        print("Please run: pip install pyyaml\n")
        sys.exit(1)
        
    try:
        import seaborn as sns
    except ImportError:
        print("\nERROR: Missing 'seaborn' library.")
        print("Please run: pip install seaborn\n")
        sys.exit(1)
        
    main()