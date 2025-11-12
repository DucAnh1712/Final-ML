# feature_engineering.py (PHIÊN BẢN V3 - HCM HYBRID)
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'datetime' not in df.columns:
                df = df.reset_index()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime', drop=False)
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['day_of_week'] = df.index.dayofweek
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        return df.drop(columns=['month', 'day_of_year', 'day_of_week'])

class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        if 'sunrise' in df.columns and 'sunset' in df.columns:
            sr = pd.to_datetime(df['sunrise'], errors='coerce')
            ss = pd.to_datetime(df['sunset'], errors='coerce')
            valid_times = sr.notna() & ss.notna()
            df['daylight_hours'] = np.nan
            df.loc[valid_times, 'daylight_hours'] = (ss[valid_times] - sr[valid_times]).dt.total_seconds() / 3600
            if 'solarenergy' in df.columns:
                df['solar_per_hour'] = df['solarenergy'] / (df['daylight_hours'] + 1e-6)
        if 'tempmax' in df.columns and 'tempmin' in df.columns:
            df['temp_range'] = df['tempmax'] - df['tempmin']
        if 'temp' in df.columns and 'dew' in df.columns:
            df['dewpoint_depression'] = df['temp'] - df['dew']
        if 'sealevelpressure' in df.columns:
            df['sealevelpressure_change'] = df['sealevelpressure'].diff().fillna(0)
        return df

class ColumnPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # ✅ SỬA: Giữ lại các features quan trọng cho HCM (mưa/ẩm)
        self.feature_cols = [
            'humidity', 'sealevelpressure', 'dew', 'cloudcover', 
            'solarradiation', 'visibility', 'windspeed', 'windgust', 'precip',
            'temp' 
        ]
        self.derived_cols = [
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'daylight_hours', 'solar_per_hour', 'temp_range',
            'dewpoint_depression', 'sealevelpressure_change'
        ]
        self.final_cols = []
        pass

    def fit(self, X, y=None):
        all_cols_available = list(X.columns)
        existing_features = [col for col in self.feature_cols if col in all_cols_available]
        existing_derived = [col for col in self.derived_cols if col in all_cols_available]
        
        self.final_cols = existing_features + existing_derived
        
        if 'temp' in self.final_cols:
             self.final_cols.remove('temp') 
        return self

    def transform(self, X):
        df = X[self.final_cols].copy()
        
        # ✅ SỬA LỖI LEAKAGE: Chỉ dùng ffill() và fillna(0)
        df = df.fillna(method='ffill')
        df = df.fillna(0) 
        
        return df

def create_feature_pipeline():
    """
    Tạo pipeline feature engineering MỚI (V3 - HCM)
    """
    feature_pipeline = Pipeline([
        ('add_time_features', TimeFeatureTransformer()),
        ('add_derived_features', DerivedFeatureTransformer()),
        ('preprocess_columns', ColumnPreprocessor())
    ])
    
    return feature_pipeline