# visualize_results.py
import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import config
import sys

def load_metrics_to_dataframe():
    """Load both metrics files (train and test) and merge them"""
    
    # 1. Define paths
    train_metrics_path = os.path.join(config.OUTPUT_DIR, config.TRAIN_METRICS_NAME)
    # (inference_linear.py saves this file)
    test_metrics_path = os.path.join(config.OUTPUT_DIR, "test_metrics_linear.yaml")

    # 2. Check files
    if not os.path.exists(train_metrics_path):
        print(f"❌ Error: {config.TRAIN_METRICS_NAME} not found. Please run train_linear.py.")
        return None
    if not os.path.exists(test_metrics_path):
        print(f"❌ Error: test_metrics_linear.yaml not found. Please run inference_linear.py.")
        return None

    # 3. Load 2 YAML files
    try:
        with open(train_metrics_path, 'r') as f:
            train_data = yaml.safe_load(f)
        with open(test_metrics_path, 'r') as f:
            test_data = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error reading YAML file: {e}")
        print("Please ensure you have PyYAML installed: pip install pyyaml")
        return None

    # 4. Convert to DataFrame
    df_train = pd.DataFrame.from_dict(train_data, orient='index')
    df_test = pd.DataFrame.from_dict(test_data, orient='index')

    # 5. Join 2 DataFrames
    df = df_train.join(df_test, lsuffix='_train', rsuffix='_test')

    # 6. Create Horizon column (1, 2, 3... 7) for plotting
    df['Horizon'] = df.index.str.replace('target_t', '').astype(int)
    df = df.sort_values('Horizon')
    
    print("✅ Loaded and merged 2 metrics files:")
    print(df[['RMSE_train', 'RMSE_test', 'R2_train', 'R2_test']])
    return df

def plot_rmse(df):
    """Plot RMSE (Train vs Test) by Horizon"""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Draw 2 lines
    plt.plot(df['Horizon'], df['RMSE_train'], marker='o', linestyle='--', label='Train RMSE (On 85% data)')
    plt.plot(df['Horizon'], df['RMSE_test'], marker='o', linestyle='-', label='Test RMSE (On 15% unseen data)', linewidth=2.5)

    plt.title('Model Performance (RMSE) by Forecast Horizon', fontsize=16)
    plt.xlabel('Forecast Day (T+N)', fontsize=12)
    plt.ylabel('RMSE (Temperature Error °C)', fontsize=12)
    plt.xticks(df['Horizon']) # Ensure all horizons (1-7) are shown
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Save file
    plot_path_rmse = os.path.join(config.OUTPUT_DIR, 'rmse_by_horizon.png')
    plt.savefig(plot_path_rmse)
    print(f"💾 Saved RMSE plot: {plot_path_rmse}")

def plot_r2(df):
    """Plot R² (Train vs Test) by Horizon"""
    # Need to "melt" dataframe for grouped bar plot
    df_melted = df.melt(id_vars=['Horizon'], 
                        value_vars=['R2_train', 'R2_test'], 
                        var_name='Metric', 
                        value_name='R-Squared')

    # Rename metric for clarity
    df_melted['Metric'] = df_melted['Metric'].map({'R2_train': 'Train R²', 'R2_test': 'Test R²'})

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Horizon', y='R-Squared', hue='Metric', palette="coolwarm")

    plt.title('Model Fit (R²) (Train vs Test) by Horizon', fontsize=16)
    plt.xlabel('Forecast Day (T+N)', fontsize=12)
    plt.ylabel('R-Squared (R²)', fontsize=12)
    plt.ylim(0, 1.0) # R² is only from 0 to 1
    plt.legend(title="Metric", loc='upper right')
    plt.grid(True, axis='y', alpha=0.5)

    # Save file
    plot_path_r2 = os.path.join(config.OUTPUT_DIR, 'r2_by_horizon.png')
    plt.savefig(plot_path_r2)
    print(f"💾 Saved R2 plot: {plot_path_r2}")

def main():
    df_metrics = load_metrics_to_dataframe()
    if df_metrics is not None:
        plot_rmse(df_metrics)
        plot_r2(df_metrics)
        print("\n🎉 VISUALIZATION COMPLETE!")
        print(f"See the 2 .png files in: {config.OUTPUT_DIR}")

if __name__ == "__main__":
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