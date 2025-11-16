import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    ✅ NÂNG CẤP (HOURLY): Tạo cyclical time features (sin/cos encoding)
    cho cả chu kỳ năm (yearly) và ngày (daily).
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # 1. Đảm bảo có DatetimeIndex
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'datetime' not in df.columns:
                df = df.reset_index()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime', drop=False)
        
        # 2. Đặc trưng chu kỳ năm (Giữ nguyên)
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['day_of_week'] = df.index.dayofweek
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # 3. ✅ THÊM MỚI (HOURLY): Đặc trưng chu kỳ 24 giờ
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 4. Trả về
        return df.drop(columns=['month', 'day_of_year', 'day_of_week', 'hour'])


class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    ✅ NÂNG CẤP (HOURLY): Tạo derived features,
    thêm các đặc trưng trễ (lag) và trượt (rolling) 24 giờ.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Các đặc trưng row-wise (Giữ nguyên)
        if 'sunrise' in df.columns and 'sunset' in df.columns:
            sr = pd.to_datetime(df['sunrise'], errors='coerce')
            ss = pd.to_datetime(df['sunset'], errors='coerce')
            valid_times = sr.notna() & ss.notna()
            df['daylight_hours'] = np.nan
            df.loc[valid_times, 'daylight_hours'] = (
                (ss[valid_times] - sr[valid_times]).dt.total_seconds() / 3600
            )
            if 'solarenergy' in df.columns:
                df['solar_per_hour'] = df['solarenergy'] / (df['daylight_hours'] + 1e-6)
        
        if 'tempmax' in df.columns and 'tempmin' in df.columns:
            df['temp_range'] = df['tempmax'] - df['tempmin']
        
        if 'temp' in df.columns and 'dew' in df.columns:
            df['dewpoint_depression'] = df['temp'] - df['dew']
        
        if 'sealevelpressure' in df.columns:
            # .diff(1) là thay đổi so với 1 GIỜ trước
            df['sealevelpressure_change'] = df['sealevelpressure'].diff() 
        
        # ✅ THÊM MỚI (HOURLY): Đặc trưng trễ 24 giờ (cực kỳ quan trọng)
        # Lấy giá trị của 24 giờ trước
        df['temp_lag_24h'] = df['temp'].shift(24)
        df['humidity_lag_24h'] = df['humidity'].shift(24)
        
        # ✅ THÊM MỚI (HOURLY): Đặc trưng trượt (Rolling)
        # shift(1) để tránh rò rỉ (leakage) dữ liệu của giờ hiện tại
        # min_periods=1 để xử lý các NaN ở 23 giờ đầu tiên
        df['temp_rolling_avg_24h'] = df['temp'].shift(1).rolling(24, min_periods=1).mean()
        df['precip_rolling_sum_24h'] = df['precip'].shift(1).rolling(24, min_periods=1).sum()
        
        # fillna(0) cho các đặc trưng .diff() và .shift()
        cols_to_fill = ['sealevelpressure_change', 'temp_lag_24h', 
                        'humidity_lag_24h', 'temp_rolling_avg_24h',
                        'precip_rolling_sum_24h']
        
        for col in cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                
        return df


class ColumnPreprocessor(BaseEstimator, TransformerMixin):
    """
    ✅ NÂNG CẤP (HOURLY): Cập nhật danh sách cột và sửa cú pháp fillna.
    """
    def __init__(self):
        # 1. Core weather features (Giữ nguyên)
        self.feature_cols = [
            'humidity', 'sealevelpressure', 'dew', 'cloudcover', 
            'solarradiation', 'visibility', 'windspeed', 'windgust', 'precip',
            'temp'  # Sẽ bị remove sau (tránh target leakage)
        ]
        
        # 2. ✅ CẬP NHẬT: Danh sách derived features
        self.derived_cols = [
            # Time (Yearly)
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            # Time (Hourly)
            'hour_sin', 'hour_cos', 
            # Row-wise
            'daylight_hours', 'solar_per_hour', 'temp_range',
            'dewpoint_depression', 'sealevelpressure_change',
            # Lag 24h
            'temp_lag_24h', 'humidity_lag_24h',
            # Rolling 24h
            'temp_rolling_avg_24h', 'precip_rolling_sum_24h'
        ]
        
        self.final_cols = []
    
    def fit(self, X, y=None):
        """Determine which columns are available"""
        all_cols_available = list(X.columns)
        existing_features = [col for col in self.feature_cols if col in all_cols_available]
        existing_derived = [col for col in self.derived_cols if col in all_cols_available]
        
        self.final_cols = existing_features + existing_derived
        
        # ✅ Remove 'temp' to avoid target leakage (Giữ nguyên)
        if 'temp' in self.final_cols:
            self.final_cols.remove('temp')
        
        return self
    
    def transform(self, X):
        """Select columns and handle missing values"""
        df = X[self.final_cols].copy()
        
        # ✅ SỬA CÚ PHÁP (Hết cảnh báo FutureWarning)
        df = df.ffill() # Lấp các NaN bằng giá trị phía trước
        df = df.bfill() # Lấp các NaN còn lại (ở đầu) bằng giáTtrị phía sau
        
        # Lưới an toàn cuối cùng (nếu toàn bộ cột là NaN)
        df = df.fillna(0)
        
        return df


def create_feature_pipeline():
    """
    Tạo pipeline feature engineering hoàn chỉnh (Giữ nguyên)
    
    Pipeline steps:
    1. TimeFeatureTransformer - Tạo cyclical time features (ĐÃ NÂNG CẤP)
    2. DerivedFeatureTransformer - Tạo derived features (ĐÃ NÂNG CẤP)
    3. ColumnPreprocessor - Select columns và impute (ĐÃ NÂNG CẤP)
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
    """
    feature_pipeline = Pipeline([
        ('add_time_features', TimeFeatureTransformer()),
        ('add_derived_features', DerivedFeatureTransformer()),
        ('preprocess_columns', ColumnPreprocessor())
    ])
    
    return feature_pipeline