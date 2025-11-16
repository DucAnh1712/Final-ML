# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import yaml
import config # Import from config.py

st.set_page_config(page_title="Hanoi Temperature Forecast", layout="wide")

# ======================================================
# 1. LOAD MODEL AND DATA
# ======================================================

@st.cache_resource
def load_model():
    """Load the complete pipeline (cached)."""
    model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run train.py first.")
        return None
    model = joblib.load(model_path)
    return model

@st.cache_data
def load_data_and_results():
    """Load test data and prediction results (if available)."""
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv")
    pred_path = os.path.join(config.OUTPUT_DIR, "test_predictions.csv")
    metrics_path = os.path.join(config.OUTPUT_DIR, "test_metrics.yaml")
    
    if not os.path.exists(test_path):
        st.error(f"data_test.csv not found at {test_path}. Please run data_processing.py.")
        return None, None, None
        
    df_test = pd.read_csv(test_path, parse_dates=["datetime"])
    
    df_pred = None
    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path, parse_dates=["datetime"])
        
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = yaml.safe_load(f)
            
    return df_test, df_pred, metrics

# Load resources
model = load_model()
df_test, df_pred, metrics = load_data_and_results()

# ======================================================
# 2. BUILD THE UI (Step 6)
# ======================================================

st.title("☀️ HCM Temperature Forecast Dashboard 🌧️")
st.write("Demo for Machine Learning I project, using XGBoost and Optuna.")

if model is None or df_test is None:
    st.warning("Please run the `data_processing.py`, `feature_engineering.py`, and `train.py` scripts before running the app.")
else:
    tab1, tab2 = st.tabs(["📊 Test Set Performance", "🔮 Next Day Forecast"])

    with tab1:
        st.header("Test Set Performance Evaluation")
        
        if metrics:
            st.subheader("Evaluation Metrics (Step 5)")
            cols = st.columns(4)
            cols[0].metric("RMSE", f"{metrics.get('RMSE', 0):.3f}°C")
            cols[1].metric("MAE", f"{metrics.get('MAE', 0):.3f}°C")
            cols[2].metric("R2 Score", f"{metrics.get('R2', 0):.3f}")
            cols[3].metric("MAPE", f"{metrics.get('MAPE', 0) * 100:.2f}%")
        else:
            st.info("`test_metrics.yaml` not found. Please run `inference.py`.")

        if df_pred is not None:
            st.subheader("Actual vs. Predicted Comparison Chart")
            df_plot = df_pred.dropna(subset=['predicted_temp'])
            
            # Create chart
            chart_data = df_plot.melt(
                id_vars=['datetime'], 
                value_vars=[config.TARGET_COL, 'predicted_temp'], 
                var_name='Type', 
                value_name='Temperature'
            )
            chart_data['Type'] = chart_data['Type'].map({
                config.TARGET_COL: 'Actual',
                'predicted_temp': 'Predicted'
            })

            st.line_chart(
                chart_data,
                x='datetime',
                y='Temperature',
                color='Type',
                height=500
            )
        else:
            st.info("`test_predictions.csv` not found. Please run `inference.py`.")

    with tab2:
        st.header("Next Day Forecast (Using latest test data)")
        
        # Get the last data from the test set as input
        # Need at least MAX_LAG days to create features
        last_known_data = df_test.tail(config.MAX_LAG + 1)
        
        if len(last_known_data) < config.MAX_LAG:
            st.warning(f"Need at least {config.MAX_LAG} days of data in the test set to forecast.")
        else:
            # Get the last known row
            last_actual_row = last_known_data.tail(1)
            last_actual_temp = last_actual_row[config.TARGET_COL].values[0]
            last_actual_date = last_actual_row['datetime'].dt.date.values[0]

            st.write(f"Last known actual data (Date {last_actual_date}): **{last_actual_temp:.1f}°C**")
            
            # The pipeline will automatically create features from `last_known_data`
            # and predict on all rows.
            
            predictions = model.predict(last_known_data)
            # The last prediction corresponds to the forecast for the day AFTER the last known date
            next_day_prediction = predictions[-1] 
            
            next_day_date = last_actual_date + pd.Timedelta(days=1)

            st.metric(
                label=f"Forecast for tomorrow ({next_day_date})",
                value=f"{next_day_prediction:.1f}°C"
            )
            
            st.info("""
            **Explanation:**
            - The model is trained to predict $T+1$ (tomorrow) based on data from $T$ (today) and its lags ($T-1$, $T-2$,...).
            - We pass the last N days of data (e.g., days $D-N$ to $D$) into the pipeline.
            - The pipeline (which includes `LagRollingFeatures`) automatically uses historical data ($D-1$, $D-2$...) to create features for day $D$.
            - The model then predicts the temperature for day $D+1$ (tomorrow).
            """)