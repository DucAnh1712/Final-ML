# inference.py
import os
import pandas as pd
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import config

def load_production_model():
    """Load the trained pipeline (features + scaler + model)."""
    model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")
    
    model = joblib.load(model_path)
    print(f"✅ Pipeline loaded from: {model_path}")
    return model

def load_test_data():
    """Load test data (processed, before features)."""
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ data_test.csv not found at {test_path}")
    
    df = pd.read_csv(test_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    print(f"✅ Test data loaded: {df.shape}")
    
    X_test = df.drop(columns=[config.TARGET_COL], errors='ignore')
    y_test = df[config.TARGET_COL]
    
    return X_test, y_test, df

def evaluate_on_test(model, X_test, y_test):
    """
    Evaluate model on test set (1-step-ahead forecast).
    Step 5: Use metrics RMSE, MAPE, R2
    """
    print("⚙️ Predicting on test set...")
    y_pred = model.predict(X_test)

    # Ensure y_test and y_pred have the same length
    # The pipeline might create NaNs at the start if the test set is too short,
    # but this shouldn't happen with our feature engineering setup.
    if len(y_pred) != len(y_test):
        print(f"⚠️ Warning: Prediction length ({len(y_pred)}) does not match y_test ({len(y_test)}).")
        # Assuming NaNs at the beginning
        nan_rows = len(y_test) - len(y_pred)
        if nan_rows > 0:
            y_test = y_test.iloc[nan_rows:]

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred)
    }

    print("\n📊 Test Set Performance (1-step-ahead):")
    for k, v in metrics.items():
        print(f"   {k:<6}: {v:.4f}")
        
    return y_pred, metrics

def save_results(df_test, y_pred, metrics, output_dir):
    """Save prediction results and metrics."""
    result_df = df_test.copy()
    
    # Handle length if y_pred is shorter (due to initial NaNs)
    nan_rows = len(result_df) - len(y_pred)
    preds_series = pd.Series(y_pred, index=result_df.index[nan_rows:])
    result_df["predicted_temp"] = preds_series

    pred_path = os.path.join(output_dir, "test_predictions.csv")
    metrics_path = os.path.join(output_dir, "test_metrics.yaml")

    result_df.to_csv(pred_path, index=False)
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, sort_keys=False)

    print(f"\n💾 Predictions saved to: {pred_path}")
    print(f"🧾 Metrics saved to: {metrics_path}")
    return result_df

def visualize_predictions(df_results, output_dir):
    """Plot comparison of actual vs predicted."""
    df_plot = df_results.dropna(subset=['predicted_temp'])
    y_test = df_plot[config.TARGET_COL]
    y_pred = df_plot['predicted_temp']

    plt.figure(figsize=(15, 6))
    plt.plot(df_plot['datetime'], y_test, label="Actual", marker='.', linestyle='-')
    plt.plot(df_plot['datetime'], y_pred, label="Predicted", marker='x', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Test Set Performance (Actual vs. Predicted)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    line_path = os.path.join(output_dir, "test_predictions_line_plot.png")
    plt.savefig(line_path)
    print(f"📈 Line plot saved to: {line_path}")
    # plt.show() # Disabled for script running

def main():
    model = load_production_model()
    X_test, y_test, df_test = load_test_data()
    y_pred, metrics = evaluate_on_test(model, X_test, y_test)
    df_results = save_results(df_test, y_pred, metrics, config.OUTPUT_DIR)
    visualize_predictions(df_results, config.OUTPUT_DIR)
    print("🎉 Test set evaluation complete.")

if __name__ == "__main__":
    main()