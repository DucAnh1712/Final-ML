# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import yaml
import base64
import matplotlib.pyplot as plt
from datetime import datetime
import config
from streamlit_autorefresh import st_autorefresh
# ======================================================
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="Daily Ho Chi Minh Temperature Forecast",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Tự động tải lại trang sau mỗi 10 phút (600,000 ms)
st_autorefresh(interval=600000, key="theme_refresher")
# ======================================================
# <<< KHỐI CSS NÂNG CẤP VỚI MÂY VÀ SAO (PHIÊN BẢN 3.0) >>>
# ======================================================
current_dt = datetime.now()
current_hour = current_dt.hour
is_daytime = 6 <= current_hour < 18
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DAY_BACKGROUND_PATH = os.path.join(APP_DIR, "assets", "day_sky_background.png")
DAY_BACKGROUND_DATA_URI = None
if os.path.exists(DAY_BACKGROUND_PATH):
    with open(DAY_BACKGROUND_PATH, "rb") as bg_file:
        DAY_BACKGROUND_DATA_URI = f"data:image/png;base64,{base64.b64encode(bg_file.read()).decode('utf-8')}"
bg_layer_0_size = "200% 200%"
bg_layer_0_animation = "none"
if is_daytime:
    # --- CÀI ĐẶT BAN NGÀY (THEO ẢNH BÌNH MINH) ---
    sky_icon = "☀️"
    if DAY_BACKGROUND_DATA_URI:
        css_background_gradient = f"url('{DAY_BACKGROUND_DATA_URI}') center/cover no-repeat"
        bg_layer_0_size = "cover"
        bg_layer_0_animation = "none"
    else:
        css_background_gradient = "linear-gradient(180deg, #cadeff 0%, #8fc0ff 32%, #4a91f5 65%, #1d5fd1 100%)"
    cloud_core_color = "rgba(255,255,255,0.95)"
    cloud_mid_color = "rgba(255,255,255,0.78)"
    cloud_highlight_color = "rgba(255,255,255,0.62)"
    cloud_shadow_color = "rgba(110,140,185,0.45)"
    css_layer_1 = """
        background: none;
        opacity: 0;
        animation: none;
        filter: none;
        mix-blend-mode: normal;
    """
    css_layer_2 = """
        background: none;
        opacity: 0;
        animation: none;
        mix-blend-mode: normal;
    """
    # Cài đặt thẻ (card)
    card_background = "rgba(255, 255, 255, 0.7)"
    card_text_color = "#0b2545"
    muted_text = "#4f6272"
    hero_overlay = "rgba(255, 255, 255, 0.22)"
    highlight_gradient = "linear-gradient(120deg, rgba(255,175,189,0.78) 0%, rgba(255,195,160,0.62) 100%)"
    hero_text_color = "#10253f"
    hero_metric_span_color = "rgba(11,37,69,0.7)"
    hero_metric_value_color = "#0b2545"
    forecast_temp_color = "#0b2545"
    summary_temp_color = "#0b2545"
else:
    # --- CÀI ĐẶT CHO BAN ĐÊM (CÓ SAO) ---
    sky_icon = "🌙"
    # Lớp 1: Nền galaxy
    css_background_gradient = "linear-gradient(135deg, #020111, #19193a, #0d1a2f, #19193a)"
    # Lớp 2: Sao xa (nhỏ)
    css_layer_1 = """
        background: radial-gradient(1px 1px at 20px 30px, #eee, transparent),
                    radial-gradient(1px 1px at 40px 60px, #fff, transparent),
                    radial-gradient(1px 1px at 50px 120px, #ddd, transparent),
                    radial-gradient(1px 1px at 100px 240px, #eee, transparent),
                    radial-gradient(1px 1px at 160px 300px, #fff, transparent),
                    radial-gradient(1px 1px at 220px 180px, #ddd, transparent);
        background-size: 300px 300px;
        opacity: 0.7;
        animation: none;
    """
    # Lớp 3: Sao gần (lớn)
    css_layer_2 = """
        background: radial-gradient(2px 2px at 50px 50px, #fff, transparent),
                    radial-gradient(2px 2px at 100px 100px, #eee, transparent),
                    radial-gradient(2px 2px at 150px 250px, #fff, transparent),
                    radial-gradient(2px 2px at 250px 130px, #ddd, transparent);
        background-size: 250px 250px;
        opacity: 0.8;
        animation: none;
    """
    # Cài đặt thẻ (card)
    card_background = "rgba(7, 22, 37, 0.38)"
    card_text_color = "#e0f7fa"
    muted_text = "#9fbcd4"
    hero_overlay = "rgba(3, 12, 24, 0.18)"
    highlight_gradient = "linear-gradient(120deg, rgba(95,114,190,0.82) 0%, rgba(153,33,232,0.64) 100%)"
    forecast_temp_color = "#e9f5ff"
    summary_temp_color = "#e9f5ff"
    hero_text_color = "#ffffff"
    hero_metric_span_color = "rgba(255,255,255,0.75)"
    hero_metric_value_color = "#e9f5ff"
    bg_layer_0_size = "200% 200%"
    bg_layer_0_animation = "gradientAnimation 45s ease infinite"
# --- TIÊM CSS VÀ HTML CHO CÁC LỚP BACKGROUND ---
st.markdown(f"""
<style>
    /* --- ĐỊNH NGHĨA ANIMATION --- */
    @keyframes gradientAnimation {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    /* --- ÁP DỤNG CSS --- */
    /* Lớp CSS chung cho các layer background */
    .background-layer {{
        position: fixed; /* Fix cứng với viewport */
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden; /* Tránh tràn */
    }}
    /* Lớp 1: Nền Gradient */
    #bg-layer-0 {{
        background: {css_background_gradient};
        background-size: {bg_layer_0_size};
        animation: {bg_layer_0_animation};
        z-index: -3; /* Lớp dưới cùng */
    }}
    /* Lớp 2: Mây Xa / Sao Xa */
    #bg-layer-1 {{
        {css_layer_1}
        z-index: -2;
    }}
    /* Lớp 3: Mây Gần / Sao Gần */
    #bg-layer-2 {{
        {css_layer_2}
        z-index: -1;
    }}
    /* --- CSS CHO CÁC THÀNH PHẦN CÒN LẠI --- */
    /* Đảm bảo .stApp TRONG SUỐT để thấy các layer bên dưới */
    .stApp {{
        background: transparent !important; 
        color: {card_text_color};
    }}
    /* <<< SỬA ĐỔI: LÀM TRONG SUỐT SIDEBAR >>> */
    [data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(180deg, rgba(8,15,32,0.95), rgba(23,38,68,0.92));
        backdrop-filter: blur(12px);
        color: #f5f9ff;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 24px 55px rgba(0, 0, 0, 0.35);
        border-radius: 0 24px 24px 0;
    }}
    /* Đổi màu chữ mặc định cho các thành phần trong sidebar */
    .stSidebar [data-testid="stMarkdownContainer"] p {{
        color: #f5f9ff !important;
    }}
    /* <<< KẾT THÚC SỬA ĐỔI SIDEBAR >>> */
    .main-header {{
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #f5f9ff;
        text-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        padding: 1rem 0;
    }}
    .metric-card {{
        background-color: {card_background};
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        backdrop-filter: blur(12px);
        color: {card_text_color};
    }}
    .forecast-box {{
        background: {highlight_gradient};
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }}
    .info-box {{
        background-color: {card_background};
        padding: 1rem 1.5rem;
        border-left: 4px solid rgba(33,150,243,0.7);
        border-radius: 10px;
            /* --- TAB 4 (MODEL DETAILS) HIGH-CONTRAST THEME --- */
            div[data-baseweb="tab-panel"]:nth-of-type(4) {{
                color: #f5f9ff;
                background: linear-gradient(135deg, rgba(8,12,28,0.88), rgba(23,40,72,0.78));
                border-radius: 22px;
                padding: 1.5rem 2rem;
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: 0 28px 60px rgba(0,0,0,0.35);
            }}
            div[data-baseweb="tab-panel"]:nth-of-type(4) h1,
                color: #f5f9ff;
            div[data-baseweb="tab-panel"]:nth-of-type(4) li,
            div[data-baseweb="tab-panel"]:nth-of-type(4) span,
            div[data-baseweb="tab-panel"]:nth-of-type(4) label {{
                color: #f5f9ff !important;
                text-shadow: 0 0 8px rgba(0,0,0,0.55);
            }}
            div[data-baseweb="tab-panel"]:nth-of-type(4) [data-testid="stDataFrame"] {{
                background: rgba(10, 18, 35, 0.85);
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 14px;
                box-shadow: 0 18px 35px rgba(0,0,0,0.45);
                color: #f5f9ff !important;
                text-shadow: 0 0 8px rgba(0,0,0,0.55);
                font-weight: 600;
                color: #f5f9ff !important;
            }}
            /* <<< SỬA ĐỔI: LÀM TRONG SUỐT SIDEBAR >>> */
        color: {card_text_color};
    }}
    .weather-hero {{
        position: relative;
        border-radius: 24px;
        padding: 2rem 3rem;
        margin: 1.5rem 0;
        background: {highlight_gradient};
        color: {hero_text_color};
        overflow: hidden;
        box-shadow: 0 25px 45px rgba(0,0,0,0.18);
    }}
    .weather-hero::after {{
        content: '';
        position: absolute;
        inset: 0;
        background: {hero_overlay};
        z-index: -1; /* Nằm dưới nội dung hero */
    }}
    .weather-hero-content {{
        position: relative;
        display: flex;
        justify-content: space-between;
        gap: 2rem;
        flex-wrap: wrap;
        align-items: center;
    }}
    .hero-metric span {{
        font-size: 0.9rem;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        color: {hero_metric_span_color};
    }}
    .hero-metric strong {{
        display: block;
        font-size: 2.5rem;
        margin-top: 0.4rem;
        color: {hero_metric_value_color};
    }}
    .forecast-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1.2rem;
        margin: 1.5rem 0;
    }}
    .forecast-card {{
        background: {card_background};
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 16px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(14px);
        color: {card_text_color};
        border: 1px solid rgba(255,255,255,0.15);
    }}
    .forecast-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }}
    .forecast-temp {{
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: {forecast_temp_color};
    }}
    .forecast-meta {{
        font-size: 0.95rem;
        color: {muted_text};
        margin-top: 0.5rem;
    }}
    .trend-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        background: rgba(255,255,255,0.32);
        color: {card_text_color};
    }}
    .forecast-summary {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
    }}
    .summary-card {{
        background: {card_background};
        border-radius: 16px;
        padding: 1rem 1.5rem;
        box-shadow: 0 12px 25px rgba(0,0,0,0.08);
        backdrop-filter: blur(12px);
    }}
</style>
<div class="background-layer" id="bg-layer-0"></div>
<div class="background-layer" id="bg-layer-1"></div>
<div class="background-layer" id="bg-layer-2"></div>
""", unsafe_allow_html=True)
# ======================================================
# LOAD FUNCTIONS
# (Phần này giữ nguyên)
# ======================================================
@st.cache_resource
def load_all_models():
    """Load all trained models for T+1 to T+7."""
    models = {}
    for target_name in config.TARGET_FORECAST_COLS: #
        model_name = f"{target_name}_model_linear.pkl" #
        model_path = os.path.join(config.MODEL_DIR, model_name) #
        if os.path.exists(model_path):
            models[target_name] = joblib.load(model_path)
    return models if models else None
@st.cache_data
def load_data_and_results():
    """Load test data and prediction results."""
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv") #
    pred_path = os.path.join(config.OUTPUT_DIR, "test_predictions_linear.csv") #
    metrics_path = os.path.join(config.OUTPUT_DIR, "test_metrics_linear.yaml") #
    df_test = None
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path, parse_dates=["datetime"])
    df_pred = None
    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path)
        # Handle duplicate datetime columns
        cols = df_pred.columns.tolist()
        if cols[0] == 'datetime' and cols[1] == 'datetime':
            df_pred = df_pred.iloc[:, 1:]  # Drop first column
        df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = yaml.load(f, Loader=yaml.UnsafeLoader)
    return df_test, df_pred, metrics
@st.cache_data
def load_best_params():
    """Load best hyperparameters from Optuna tuning."""
    params_path = os.path.join(config.MODEL_DIR, "optuna_best_params_linear.yaml") #
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)
    return None
@st.cache_data
def load_train_metrics():
    """Load training metrics."""
    metrics_path = os.path.join(config.OUTPUT_DIR, "train_metrics_linear.yaml") #
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)
    return None
# ======================================================
# LOAD RESOURCES
# ======================================================
models = load_all_models()
df_test, df_pred, test_metrics = load_data_and_results()
best_params = load_best_params()
train_metrics = load_train_metrics()
forecast_horizon_labels = ", ".join([f"T+{h}" for h in config.FORECAST_HORIZONS])
def get_weather_icon(temp: float) -> str:
    """Return a weather emoji based on the temperature value."""
    if temp >= 35:
        return "🔥"
    if temp >= 31:
        return "☀️"
    if temp >= 28:
        return "🌤️"
    if temp >= 25:
        return "⛅"
    if temp >= 22:
        return "🌥️"
    return "🌧️"
# ======================================================
# SIDEBAR
# (Giữ nguyên)
# ======================================================
with st.sidebar:
    #st.image("https://img.icons8.com/clouds/200/000000/weather.png", width=150)
    st.markdown("<h3 style='color:#f5f9ff; margin-bottom:0.4rem;'>🌡️ Ho Chi Minh Weather Forecast</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📊 Project Infor**")
    st.markdown(
        f"""
        - **Location**: Ho Chi Minh City
        - **Forecast Horizons**: {forecast_horizon_labels}
        - **Latest Batch**: {current_dt.strftime('%d %b %Y')}
        - **Models**: Linear family (Linear/Ridge/Lasso/ElasticNet)
        """
    )
    # st.markdown("**🛠 Pipeline (`daily/src`)**")
    # st.markdown(
    #     """
    #     1. `python daily/src/data_processing.py`
    #     2. `python daily/src/train.py`
    #     3. `python daily/src/inference.py`
    #     """
    # )
    st.markdown("**📦 Feature Stack**")
    st.markdown(
        """
        - Cyclical month/day sine-cosine encodings
        - Daylight, solar-per-hour, temp-range, dewpoint spread
        - Core sensors: humidity, pressure, wind, precip, radiation
        """
    )
    st.markdown("---")
    st.markdown("**👥 Group 6 - Machine Learning Final Project**") #
# ======================================================
# MAIN HEADER
# ======================================================
st.markdown(f'<h1 class="main-header">{sky_icon} Ho Chi Minh Temperature Forecast Dashboard 🌧️</h1>', 
            unsafe_allow_html=True)
subtitle_color = "#4f6272" if is_daytime else "#d0e6ff"
# st.markdown(
#     f"<p style='text-align: center; color: {subtitle_color}; font-size: 1.1rem;'>Daily-resolution pipeline (time + derived features feeding a linear stack) for Ho Chi Minh City • Last refresh {current_dt.strftime('%d %b %Y')}</p>",
#     unsafe_allow_html=True)
# ======================================================
# CHECK IF DATA IS AVAILABLE
# (Giữ nguyên)
# ======================================================
if models is None or df_test is None:
    st.error("⚠️ **Models or data not found!**")
    st.info("""
    Please run the following scripts in order:
    1. `python daily/src/data_processing.py` - Process raw data
    2. `python daily/src/optuna_search_linear.py` - Run Optuna tuning
    3. `python daily/src/train.py` - Train models
    4. `python daily/src/inference.py` - Generate predictions
    """)
    st.stop()
# ======================================================
# TABS
# (Giữ nguyên)
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Multi-Target Forecast", 
    "📊 Model Performance", 
    "📈 Visualizations",
    "⚙️ Model Details"
])
# ======================================================
# TAB 1: MODEL PERFORMANCE
# (Giữ nguyên)
# ======================================================
with tab1:
    st.header("🔮 Multi-Target Temperature Forecast")
    st.markdown('<div class="info-box">💡 <b>How it works:</b>  Forecasts refresh automatically based on the latest inference results. We report T+1 to T+7 day horizons with stylish weather cards.</div>', 
                unsafe_allow_html=True)
    # Get latest data for forecasting
    if len(df_test) > 0:
        window_size = max(getattr(config, 'WINDOWS', [30])) if hasattr(config, 'WINDOWS') else 30
        last_known_data = df_test.tail(window_size + 10).copy()
        last_row = df_test.iloc[-1]
        last_actual_temp = last_row[config.TARGET_COL] #
        last_actual_date = last_row['datetime']
        # Use test data predictions instead of trying to predict from raw data
        if df_pred is not None and len(df_pred) > 0:
            last_pred_row = df_pred.iloc[-1]
            forecasts = {}
            forecast_dates = {}
            for target_name in config.TARGET_FORECAST_COLS: #
                pred_col = f'pred_{target_name}'
                if pred_col in df_pred.columns:
                    forecast_value = last_pred_row[pred_col]
                    days_ahead = int(target_name.split('_t')[-1])
                    forecast_date = last_actual_date + pd.Timedelta(days=days_ahead)
                    forecasts[target_name] = forecast_value
                    forecast_dates[target_name] = forecast_date
        else:
            forecasts = {}
            forecast_dates = {}
            st.warning("⚠️ No prediction data available. Please run `python daily/src/inference.py` first.")
        if forecasts:
            sorted_targets = sorted(forecasts.keys(), key=lambda t: int(t.split('_t')[-1]))
            filtered_targets = sorted_targets[:7]
            forecast_temps = [forecasts[t] for t in filtered_targets]
            avg_forecast = np.mean(forecast_temps)
            max_target = max(filtered_targets, key=lambda t: forecasts[t])
            min_target = min(filtered_targets, key=lambda t: forecasts[t])
            time_label = "Good morning" if current_hour < 12 else ("Good afternoon" if current_hour < 18 else "Good evening")
            hero_html = f"""
            <div class="weather-hero">
                <div class="weather-hero-content">
                    <div>
                        <p style="margin:0; letter-spacing:0.2rem; text-transform:uppercase; opacity:0.8;">{time_label}</p>
                        <h2 style="margin:0.4rem 0 0; font-size:2.2rem;">{sky_icon} Live Forecast Center</h2>
                        <p style="opacity:0.85; margin-top:0.5rem;">Latest observation: {last_actual_date.strftime("%B %d, %Y")} • {len(last_known_data)} recent samples</p>
                        <p style="opacity:0.8; margin-top:0.3rem;">Forecast range: next 7 days</p>
                    </div>
                    <div class="hero-metric">
                        <span>Current Temperature</span>
                        <strong>{last_actual_temp:.1f}°C</strong>
                    </div>
                    <div class="hero-metric">
                        <span>Outlook Avg</span>
                        <strong>{avg_forecast:.1f}°C</strong>
                    </div>
                </div>
            </div>
            """
            st.markdown(hero_html, unsafe_allow_html=True)
            # Forecast cards grid rendered as a single row of up to 7 columns
            cols = st.columns(len(filtered_targets))
            for col, target_name in zip(cols, filtered_targets):
                temp = forecasts[target_name]
                forecast_date = forecast_dates[target_name]
                days_ahead = int(target_name.split('_t')[-1])
                delta = temp - last_actual_temp
                trend_symbol = "▲" if delta >= 0 else "▼"
                trend_text = f"{trend_symbol} {abs(delta):.1f}° vs today"
                icon = get_weather_icon(temp)
                day_name = forecast_date.strftime("%A")
                col.markdown(f"""
                <div class='forecast-card'>
                    <div class='forecast-icon'>{icon}</div>
                    <p style='margin:0; font-weight:600;'>{day_name}</p>
                    <h3 class='forecast-temp'>{temp:.1f}°C</h3>
                    <div class='forecast-meta'>{forecast_date.strftime("%d %b %Y")}</div>
                    <div class='forecast-meta'>Feels like trend</div>
                    <span class='trend-pill'>{trend_text}</span>
                </div>
                """, unsafe_allow_html=True)
            # Visualization
            st.markdown("### 📈 Forecast Timeline")
            dates = [last_actual_date] + [forecast_dates[t] for t in filtered_targets]
            temps = [last_actual_temp] + [forecasts[t] for t in filtered_targets]
            # Round temperatures to 0.1°C so equal displayed values plot as flat
            temps_plot = [round(float(x), 1) if pd.notna(x) else np.nan for x in temps]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot([dates[0]], [temps_plot[0]], 'o', markersize=12, color='black', label='Latest Known', zorder=5)
            ax.plot(dates, temps_plot, 'o-', linewidth=2.5, markersize=10, color='#FF6B6B', label='Forecast')
            for date, temp_v in zip(dates, temps_plot):
                ax.annotate(f'{temp_v:.1f}°C', xy=(date, temp_v), xytext=(0, 10), textcoords='offset points',
                            ha='center', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            ax.set_title('Multi-Step Temperature Forecast', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            # Summary tiles
            hottest_temp = forecasts[max_target]
            coolest_temp = forecasts[min_target]
            hottest_date = forecast_dates[max_target].strftime("%b %d")
            coolest_date = forecast_dates[min_target].strftime("%b %d")
            spread = hottest_temp - coolest_temp
            summary_html = f"""
            <div class='forecast-summary'>
                <div class='summary-card'>
                    <p style='margin:0; text-transform:uppercase; font-size:0.8rem; opacity:0.7;'>Average Outlook</p>
                    <h3 style='margin:0.2rem 0 0; font-size:2rem; color:{summary_temp_color};'>{avg_forecast:.1f}°C</h3>
                    <p style='margin:0.3rem 0 0;'>Stable comfort range expected</p>
                </div>
                <div class='summary-card'>
                    <p style='margin:0; text-transform:uppercase; font-size:0.8rem; opacity:0.7;'>Warmest Point</p>
                    <h3 style='margin:0.2rem 0 0; font-size:2rem; color:{summary_temp_color};'>{hottest_temp:.1f}°C</h3>
                    <p style='margin:0.3rem 0 0;'>Arriving {hottest_date} (T+{int(max_target.split('_t')[-1])})</p>
                </div>
                <div class='summary-card'>
                    <p style='margin:0; text-transform:uppercase; font-size:0.8rem; opacity:0.7;'>Coolest Point</p>
                    <h3 style='margin:0.2rem 0 0; font-size:2rem; color:{summary_temp_color};'>{coolest_temp:.1f}°C</h3>
                    <p style='margin:0.3rem 0 0;'>Arriving {coolest_date} (T+{int(min_target.split('_t')[-1])})</p>
                </div>
                <div class='summary-card'>
                    <p style='margin:0; text-transform:uppercase; font-size:0.8rem; opacity:0.7;'>Temperature Spread</p>
                    <h3 style='margin:0.2rem 0 0; font-size:2rem; color:{summary_temp_color};'>{spread:.1f}°C</h3>
                    <p style='margin:0.3rem 0 0;'>Range across the next 7 days</p>
                </div>
            </div>
            """
            st.markdown("### ☁️ Forecast Highlights")
            st.markdown(summary_html, unsafe_allow_html=True)
        else:
            st.error("Unable to generate forecasts. Please check if models are loaded correctly.")
    else:
        st.warning("No test data available for forecasting.")
# ======================================================
# TAB 2: MULTI-TARGET FORECAST
# (Giữ nguyên)
# ======================================================
with tab2:
    st.header("📊 Model Performance Evaluation")
    if test_metrics:
        st.subheader("🎯 Test Set Metrics (All Targets)")
        # Create metrics dataframe for comparison
        metrics_df = pd.DataFrame(test_metrics).T
        metrics_df.index.name = 'Target'
        metrics_df = metrics_df.reset_index()
        # Display metrics in columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📉 Error Metrics")
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(metrics_df))
            width = 0.35
            ax.bar(x - width/2, metrics_df['RMSE'], width, label='RMSE', color='#FF6B6B')
            ax.bar(x + width/2, metrics_df['MAE'], width, label='MAE', color='#4ECDC4')
            ax.set_xlabel('Target', fontsize=12, fontweight='bold')
            ax.set_ylabel('Error (°C)', fontsize=12, fontweight='bold')
            ax.set_title('RMSE & MAE Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_df['Target'])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        with col2:
            st.markdown("### 📊 R² Score")
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#667eea', '#764ba2', '#f093fb'] # Thêm màu cho đủ 7
            bars = ax.barh(metrics_df['Target'], metrics_df['R2'], color=colors)
            ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
            ax.set_title('Model Performance (R² Score)', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, metrics_df['R2'])):
                ax.text(value + 0.02, i, f'{value:.3f}', va='center', fontweight='bold')
            st.pyplot(fig)
        # Display detailed metrics table
        st.subheader("📋 Detailed Metrics Table")
        # Format the dataframe for better display
        display_df = metrics_df.copy()
        display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:.4f}°C")
        display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.4f}°C")
        display_df['R2'] = display_df['R2'].apply(lambda x: f"{x:.4f}")
        display_df['MAPE'] = display_df['MAPE'].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(display_df, use_container_width=True)
        # Show training vs test metrics comparison if available
        if train_metrics:
            st.subheader("🔍 Overfitting Analysis: Train vs Test")
            comparison_data = []
            for target in config.TARGET_FORECAST_COLS: #
                if target in train_metrics and target in test_metrics:
                    comparison_data.append({
                        'Target': target,
                        'Train MAE': train_metrics[target]['MAE'],
                        'Test MAE': test_metrics[target]['MAE'],
                        'Train R²': train_metrics[target]['R2'],
                        'Test R²': test_metrics[target]['R2']
                    })
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### MAE Comparison")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    x = np.arange(len(comp_df))
                    width = 0.35
                    ax.bar(x - width/2, comp_df['Train MAE'], width, label='Train', color='#95E1D3')
                    ax.bar(x + width/2, comp_df['Test MAE'], width, label='Test', color='#F38181')
                    ax.set_ylabel('MAE (°C)', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Target', fontsize=12, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(comp_df['Target'])
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    st.pyplot(fig)
                with col2:
                    st.markdown("#### R² Score Comparison")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(x - width/2, comp_df['Train R²'], width, label='Train', color='#95E1D3')
                    ax.bar(x + width/2, comp_df['Test R²'], width, label='Test', color='#F38181')
                    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Target', fontsize=12, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(comp_df['Target'])
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    st.pyplot(fig)
    else:
        st.warning("⚠️ Test metrics not found. Please run `python daily/src/inference.py` first.")
# ======================================================
# TAB 3: VISUALIZATIONS
# (Giữ nguyên)
# ======================================================
with tab3:
    st.header("📈 Prediction Visualizations")
    if df_pred is not None:
        st.subheader("🎯 Actual vs Predicted Temperature (Test Set)")
        # Create interactive plot
        fig, ax = plt.subplots(figsize=(14, 7))
        # Plot actual temperature (use target_t1 as actual since temp column may not exist)
        actual_col = config.TARGET_COL if config.TARGET_COL in df_pred.columns else 'target_t1' #
        ax.plot(df_pred['datetime'], df_pred[actual_col], 
               label='Actual', color='black', linewidth=2, alpha=0.8)
        # Plot predictions for each target
        colors_pred = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#FF9AA2', '#FFB7B2', '#FFDAC1']
        markers = ['o', 's', '^', 'd', 'v', 'p', '*']
        for idx, target_name in enumerate(config.TARGET_FORECAST_COLS): #
            pred_col = f'pred_{target_name}'
            if pred_col in df_pred.columns:
                df_plot = df_pred.dropna(subset=[pred_col])
                ax.plot(df_plot['datetime'], df_plot[pred_col], 
                       label=f'Predicted {target_name.upper()}',
                       color=colors_pred[idx], 
                       marker=markers[idx],
                       markersize=3,
                       linewidth=1.5,
                       alpha=0.7,
                       linestyle='--')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.set_title('Multi-Target Temperature Predictions on Test Set', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        # Residual plots
        st.subheader("📉 Residual Analysis")
        cols = st.columns(2)
        for idx, target_name in enumerate(config.TARGET_FORECAST_COLS[:2]):  # Show first 2
            pred_col = f'pred_{target_name}'
            if pred_col in df_pred.columns:
                with cols[idx]:
                    df_clean = df_pred.dropna(subset=[pred_col, target_name])
                    residuals = df_clean[target_name] - df_clean[pred_col]
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(df_clean[pred_col], residuals, alpha=0.5, color=colors_pred[idx])
                    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
                    ax.set_xlabel('Predicted Temperature (°C)', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Residuals (°C)', fontsize=11, fontweight='bold')
                    ax.set_title(f'Residual Plot: {target_name.upper()}', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        # Error distribution
        st.subheader("📊 Error Distribution")
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for idx, target_name in enumerate(config.TARGET_FORECAST_COLS): #
            pred_col = f'pred_{target_name}'
            if pred_col in df_pred.columns:
                df_clean = df_pred.dropna(subset=[pred_col, target_name])
                errors = df_clean[target_name] - df_clean[pred_col]
                axes[idx].hist(errors, bins=30, color=colors_pred[idx], alpha=0.7, edgecolor='black')
                axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
                axes[idx].set_xlabel('Error (°C)', fontsize=11, fontweight='bold')
                axes[idx].set_ylabel('Frequency', fontsize=11, fontweight='bold')
                axes[idx].set_title(f'{target_name.upper()} - Mean Error: {errors.mean():.3f}°C', 
                                  fontsize=12, fontweight='bold')
                axes[idx].grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("⚠️ Prediction results not found. Please run `python daily/src/inference.py` first.")
# ======================================================
# TAB 4: MODEL DETAILS
# (Giữ nguyên)
# ======================================================
with tab4:
    st.header("⚙️ Model Configuration & Hyperparameters")
    # Project configuration
    st.subheader("📋 Project Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Data Split:**")
        st.write(f"- Train Ratio: {config.TRAIN_RATIO}") #
        st.write(f"- Val Ratio: {config.VAL_RATIO}") #
        st.write(f"- Test Ratio: {1 - config.TRAIN_RATIO - config.VAL_RATIO}") #
        st.markdown("**Feature Engineering:**")
        st.write("- Time encodings: month/day sin-cos additions")
        st.write("- Derived metrics: daylight_hours, solar_per_hour, temp_range, dewpoint_depression, sealevelpressure_change")
        st.write("- Sensor inputs: humidity, pressure, dew, cloudcover, solarradiation, visibility, windspeed, windgust, precip")
        st.write(f"- Forecast Horizons: {config.FORECAST_HORIZONS}") #
    with col2:
        st.markdown("**Training Configuration:**")
        st.write(f"- Optuna Trials: {config.OPTUNA_TRIALS}") #
        st.write(f"- CV Splits: {config.CV_N_SPLITS}") #
        st.write(f"- Target Column: {config.TARGET_COL}") #
        st.markdown("**Targets:**")
        for target in config.TARGET_FORECAST_COLS: #
            st.write(f"- {target.upper()}")
    # Best hyperparameters
    if best_params:
        st.subheader("🏆 Best Hyperparameters (From Optuna)")
        if 'best_params' in best_params:
            for target_name, params in best_params['best_params'].items(): #
                with st.expander(f"📊 {target_name.upper()} - Hyperparameters"):
                    params_df = pd.DataFrame([params]).T
                    params_df.columns = ['Value']
                    params_df.index.name = 'Parameter'
                    st.dataframe(params_df, use_container_width=True)
    # Feature importance (if available)
    st.subheader("📊 Feature Information")
    st.markdown("""
     **Pipeline (`daily/src/feature_engineering.py`):**
     1. `TimeFeatureTransformer` adds cyclical encodings (month/day sin-cos).
     2. `DerivedFeatureTransformer` builds daylight_hours, solar_per_hour, temp_range,
         dewpoint_depression, and sealevelpressure_change using only same-row info.
     3. `ColumnPreprocessor` selects weather sensors (humidity, sealevelpressure, dew,
         cloudcover, solarradiation, visibility, windspeed, windgust, precip) and removes `temp`
         before forward/backward filling gaps.
    """)
# ======================================================
# FOOTER
# (Giữ nguyên)
# ======================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>Ho Chi Minh Weather Forecast Dashboard</b></p>
    <p>Group 6 - Machine Learning Final Project</p>
</div>
""", unsafe_allow_html=True)
