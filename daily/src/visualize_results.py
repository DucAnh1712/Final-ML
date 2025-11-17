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
    """
    Loads both metrics files (train and test), merges them, and prepares the data
    for visualization by adding a 'Horizon' column.
    """
    
    # 1. Define paths
    train_metrics_path = os.path.join(config.OUTPUT_DIR, config.TRAIN_METRICS_NAME)
    # The inference script is assumed to save the test metrics file
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
    horizon_values = df.index.str.replace('target_t', '').astype(int)
    
    # Simple check: if horizon values are large (e.g., 24, 48), assume hourly and convert to days
    if horizon_values.max() > 20 and horizon_values.max() % 24 == 0: 
        df['Horizon'] = horizon_values / 24
        print("💡 Assuming hourly data and converting horizon to days (N/24).")
    else: 
        # Assume daily or small integer horizons
        df['Horizon'] = horizon_values
        print("💡 Assuming daily data, horizon represents days (N).")
        
    df = df.sort_values('Horizon')
    
    print("✅ Loaded and merged 2 metrics files:")
    print(df[['Horizon', 'RMSE_train', 'RMSE_test', 'MAE_train', 'MAE_test', 'R2_train', 'R2_test']])
    return df

def plot_rmse(df):
    """Plot RMSE (Train vs Test) by Forecast Horizon."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Draw 2 lines
    plt.plot(df['Horizon'], df['RMSE_train'], marker='o', linestyle='--', label='Train RMSE (On 85% Data)')
    plt.plot(df['Horizon'], df['RMSE_test'], marker='o', linestyle='-', label='Test RMSE (On 15% Unseen Data)', linewidth=2.5)

    plt.title('Model Performance (RMSE) by Forecast Horizon', fontsize=16)
    plt.xlabel('Forecast Horizon (T+N)', fontsize=12)
    plt.ylabel('RMSE (Prediction Error)', fontsize=12)
    plt.xticks(df['Horizon']) # Ensure all horizons (1-7) are shown
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Save file
    plot_path_rmse = os.path.join(config.OUTPUT_DIR, 'rmse_by_horizon.png')
    plt.savefig(plot_path_rmse)
    plt.close() # Close plot to free memory
    print(f"💾 Saved RMSE plot: {plot_path_rmse}")

def plot_mae(df):
    """Plot MAE (Train vs Test) by Forecast Horizon."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Draw 2 lines
    plt.plot(df['Horizon'], df['MAE_train'], marker='s', linestyle='--', label='Train MAE (On 85% Data)')
    plt.plot(df['Horizon'], df['MAE_test'], marker='s', linestyle='-', label='Test MAE (On 15% Unseen Data)', linewidth=2.5)

    plt.title('Model Performance (MAE) by Forecast Horizon', fontsize=16)
    plt.xlabel('Forecast Horizon (T+N)', fontsize=12)
    plt.ylabel('MAE (Prediction Error)', fontsize=12)
    plt.xticks(df['Horizon'])
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Save file
    plot_path_mae = os.path.join(config.OUTPUT_DIR, 'mae_by_horizon.png')
    plt.savefig(plot_path_mae)
    plt.close() # Close plot to free memory
    print(f"💾 Saved MAE plot: {plot_path_mae}")


def plot_r2(df):
    """Plot R² (Train vs Test) by Forecast Horizon."""
    # Melt dataframe for grouped bar plot using seaborn
    df_melted = df.melt(id_vars=['Horizon'], 
                        value_vars=['R2_train', 'R2_test'], 
                        var_name='Metric', 
                        value_name='R-Squared')

    # Rename metric for clarity
    df_melted['Metric'] = df_melted['Metric'].map({'R2_train': 'Train R²', 'R2_test': 'Test R²'})

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Horizon', y='R-Squared', hue='Metric', palette="coolwarm")

    plt.title('Model Fit (R²) (Train vs Test) by Horizon', fontsize=16)
    plt.xlabel('Forecast Horizon (T+N)', fontsize=12)
    plt.ylabel('R-Squared (R²)', fontsize=12)
    plt.ylim(0, 1.0) # R² is typically between 0 and 1
    plt.legend(title="Metric", loc='upper right')
    plt.grid(True, axis='y', alpha=0.5)

    # Save file
    plot_path_r2 = os.path.join(config.OUTPUT_DIR, 'r2_by_horizon.png')
    plt.savefig(plot_path_r2)
    plt.close() # Close plot to free memory
    print(f"💾 Saved R2 plot: {plot_path_r2}")

def plot_overfitting_gap(df):
    """
    Plot the percentage difference (Gap) between Test RMSE and Train RMSE by Horizon.
    A positive gap indicates overfitting/generalization loss.
    """
    
    # Calculate Gap: (Test_RMSE - Train_RMSE) / Train_RMSE * 100
    df_gap = df.copy()
    df_gap['Gap (%)'] = ((df_gap['RMSE_test'] - df_gap['RMSE_train']) / df_gap['RMSE_train']) * 100
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Draw 1 line for the gap
    plt.plot(df_gap['Horizon'], df_gap['Gap (%)'], marker='x', linestyle='-', label='Overfitting Gap (Test vs Train)', color='red', linewidth=2)

    plt.title('Model Overfitting (Train-Test RMSE Gap) by Horizon', fontsize=16)
    plt.xlabel('Forecast Horizon (T+N)', fontsize=12)
    plt.ylabel('Overfitting (Gap %)', fontsize=12)
    plt.xticks(df_gap['Horizon'])
    
    # Add 0% line for reference
    plt.axhline(0, color='black', linestyle='--', linewidth=1) 
    
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Save file
    plot_path_gap = os.path.join(config.OUTPUT_DIR, 'overfitting_gap_by_horizon.png')
    plt.savefig(plot_path_gap)
    plt.close() # Close plot to free memory
    print(f"💾 Saved Overfitting Gap plot: {plot_path_gap}")

# ======================================================
# MAIN FUNCTION
# ======================================================
def main():
    """Executes the data loading and plotting sequence."""
    df_metrics = load_metrics_to_dataframe()
    if df_metrics is not None:
        plot_rmse(df_metrics)
        plot_mae(df_metrics)
        plot_r2(df_metrics)
        plot_overfitting_gap(df_metrics) 
        
        print("\n🎉 VISUALIZATION COMPLETE!")
        print(f"See the 4 .png files in the output directory: {config.OUTPUT_DIR}") 

if __name__ == "__main__":
    # Check for required libraries before starting
    try:
        import yaml
        import seaborn as sns
    except ImportError as e:
        library_name = str(e).split()[-1]
        print(f"\nERROR: Missing required library: {library_name}")
        print(f"Please run: pip install {library_name}\n")
        sys.exit(1)
        
    main()