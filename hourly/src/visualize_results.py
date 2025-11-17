import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import config
import sys

def load_one_metric_file(filepath, model_type, metric_type):
    """
    Helper function: Loads one YAML file, converts it to a DataFrame, 
    and adds columns for model type and metric type (Train/Test).
    """
    if not os.path.exists(filepath):
        print(f"❌ WARNING: Metrics file not found: {filepath}")
        print("   Run the corresponding train/inference file first.")
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        df = pd.DataFrame.from_dict(data, orient='index')
        df['model_type'] = model_type
        df['metric_type'] = metric_type
        
        # Convert 'target_t24', 'target_t48' -> 1, 2, 3... (Days)
        horizon_hours = df.index.str.replace('target_t', '').astype(int)
        df['Horizon'] = horizon_hours / 24
        
        return df
    except Exception as e:
        print(f"❌ Error reading file {filepath}: {e}")
        return None

def load_all_metrics():
    """
    Loads ALL 6 metrics files (Train/Test for 3 models: Linear, XGBoost, LightGBM).
    """
    all_dfs = []
    
    # === 1. Load Linear Model ===
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TRAIN_METRICS_LINEAR_NAME),
        'Linear', 'Train'
    ))
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TEST_METRICS_LINEAR_NAME),
        'Linear', 'Test'
    ))
    
    # === 2. Load XGBoost Model ===
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TRAIN_METRICS_XGBOOST_NAME),
        'XGBoost', 'Train'
    ))
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TEST_METRICS_XGBOOST_NAME),
        'XGBoost', 'Test'
    ))
    
    # === 3. Load LightGBM Model ===
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TRAIN_METRICS_LIGHTGBM_NAME),
        'LightGBM', 'Train'
    ))
    all_dfs.append(load_one_metric_file(
        os.path.join(config.OUTPUT_DIR, config.TEST_METRICS_LIGHTGBM_NAME),
        'LightGBM', 'Test'
    ))
    
    # Check if any file is missing
    if any(df is None for df in all_dfs):
        print("\nOne or more metrics files are missing. Halting program.")
        return None
        
    # Combine all DataFrames
    full_df = pd.concat(all_dfs)
    
    print("✅ Successfully loaded and combined 6 metrics files.")
    return full_df

# ===================================================================
# PLOTTING FUNCTIONS (UPDATED)
# ===================================================================

def plot_test_metric_comparison(df_test_only, metric_name, title, ylabel, filename, higher_is_better=False):
    """
    Generic function to plot 3 models for a single Test metric (RMSE, MAE, R2).
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Sort legend by performance (e.g., lower RMSE is better, higher R2 is better)
    sorted_models = df_test_only.groupby('model_type')[metric_name].mean().sort_values(
        ascending=higher_is_better
    ).index
    
    sns.lineplot(
        data=df_test_only,
        x='Horizon',
        y=metric_name,
        hue='model_type', # 3 colors for 3 models
        hue_order=sorted_models, # Sort legend
        style='model_type', # 3 line styles for 3 models
        style_order=sorted_models,
        markers=True,
        linewidth=2.5,
        markersize=8
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Forecast Day (T+N)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(range(1, 8)) # 1, 2, ... 7 days
    plt.legend(title="Model (Best -> Worst)")
    plt.grid(True, alpha=0.7)

    plot_path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(plot_path, dpi=120)
    print(f"💾 Chart saved: {plot_path}")

def plot_overfitting_comparison(df):
    """
    Chart 4: Compares Overfitting (Train-Test RMSE Gap) of 3 Models.
    """
    # 1. Pivot data to get Train/Test RMSE on the same row
    df_pivot = df.pivot_table(
        index=['Horizon', 'model_type'], 
        columns='metric_type', 
        values='RMSE' # Use RMSE as the basis
    ).reset_index()
    
    # 2. Calculate Gap percentage
    df_pivot['Gap (%)'] = (df_pivot['Test'] - df_pivot['Train']) / df_pivot['Train'] * 100
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 3. Plot Gap chart
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
    
    plt.title('Overfitting Comparison (Train-Test RMSE Gap) for 3 Models (Hourly)', fontsize=16, fontweight='bold')
    plt.xlabel('Forecast Day (T+N)', fontsize=12)
    plt.ylabel('Overfitting (Gap %)', fontsize=12)
    plt.xticks(range(1, 8))
    plt.legend(title="Model")
    plt.axhline(0, color='black', linestyle='--', linewidth=1) # 0% Line
    plt.grid(True, alpha=0.7)

    plot_path = os.path.join(config.OUTPUT_DIR, 'compare_ALL_MODELS_Overfitting_Gap.png')
    plt.savefig(plot_path, dpi=120)
    print(f"💾 Overfitting Gap chart saved: {plot_path}")

def main():
    # Load all 6 files
    df_full_metrics = load_all_metrics()
    
    if df_full_metrics is not None:
        # Filter Test data for reuse
        df_test_only = df_full_metrics[df_full_metrics['metric_type'] == 'Test'].copy()

        # === PLOT 4 CHARTS ===
        
        # 1. RMSE Chart (Absolute Error)
        plot_test_metric_comparison(
            df_test_only,
            metric_name='RMSE',
            title='Performance Comparison (Test RMSE) for 3 Models (Hourly)',
            ylabel='RMSE (Temperature Error °C)',
            filename='compare_ALL_MODELS_Test_RMSE.png',
            higher_is_better=False # RMSE: Lower is better
        )
        
        # 2. R2 Chart (Goodness of Fit)
        plot_test_metric_comparison(
            df_test_only,
            metric_name='R2',
            title='Goodness of Fit Comparison (Test R2) for 3 Models (Hourly)',
            ylabel='R-Squared (R²)',
            filename='compare_ALL_MODELS_Test_R2.png',
            higher_is_better=True # R2: Higher is better
        )
        
        # 3. MAE Chart (Absolute Error)
        plot_test_metric_comparison(
            df_test_only,
            metric_name='MAE',
            title='Performance Comparison (Test MAE) for 3 Models (Hourly)',
            ylabel='MAE (Temperature Error °C)',
            filename='compare_ALL_MODELS_Test_MAE.png',
            higher_is_better=False # MAE: Lower is better
        )
        
        # 4. Overfitting Chart (Using df_full_metrics)
        plot_overfitting_comparison(df_full_metrics)
        
        print("\n🎉 FINISHED VISUALIZING COMPARISON OF 3 MODELS!")
        print(f"See 4 .png files in: {config.OUTPUT_DIR}")

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