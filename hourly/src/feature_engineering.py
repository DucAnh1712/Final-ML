# hourly/src/feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    UPGRADE (HOURLY): Creates cyclical time features (sin/cos encoding) 
    for both yearly and daily cycles.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # 1. Ensure DatetimeIndex is present
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'datetime' not in df.columns:
                df = df.reset_index()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime', drop=False)
        
        # 2. Yearly cyclical features (Unchanged)
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['day_of_week'] = df.index.dayofweek
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # 3. ✅ NEW ADDITION (HOURLY): 24-hour cyclical features
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 4. Return, dropping helper columns
        return df.drop(columns=['month', 'day_of_year', 'day_of_week', 'hour'])


class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    UPGRADE (HOURLY): Creates derived features, adding 24-hour lag 
    and rolling features, crucial for hourly forecasting.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Row-wise features (Unchanged)
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
            # .diff(1) is the change compared to 1 HOUR ago
            df['sealevelpressure_change'] = df['sealevelpressure'].diff() 
        
        # ✅ NEW ADDITION (HOURLY): 24-hour lag features (critical for time series)
        # Gets the value from 24 hours ago
        if 'temp' in df.columns:
            df['temp_lag_24h'] = df['temp'].shift(24)
        if 'humidity' in df.columns:
            df['humidity_lag_24h'] = df['humidity'].shift(24)
        
        # ✅ NEW ADDITION (HOURLY): Rolling features
        # shift(1) to prevent current-hour data leakage
        # min_periods=1 to handle NaNs in the first 23 hours
        if 'temp' in df.columns:
            df['temp_rolling_avg_24h'] = df['temp'].shift(1).rolling(24, min_periods=1).mean()
        if 'precip' in df.columns:
            df['precip_rolling_sum_24h'] = df['precip'].shift(1).rolling(24, min_periods=1).sum()
        
        # fillna(0) for .diff() and .shift() features
        cols_to_fill = ['sealevelpressure_change', 'temp_lag_24h', 
                        'humidity_lag_24h', 'temp_rolling_avg_24h',
                        'precip_rolling_sum_24h']
        
        for col in cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                
        return df


class ColumnPreprocessor(BaseEstimator, TransformerMixin):
    """
    UPGRADE (HOURLY): Updates column list and fixes fillna syntax.
    """
    def __init__(self):
        # 1. Core weather features (Unchanged)
        self.feature_cols = [
            'humidity', 'sealevelpressure', 'dew', 'cloudcover', 
            'solarradiation', 'visibility', 'windspeed', 'windgust', 'precip',
            'temp'  # Will be removed later (to avoid target leakage)
        ]
        
        # 2. ✅ UPDATE: List of derived features
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
        
        # ✅ Remove 'temp' to avoid target leakage (Unchanged)
        if 'temp' in self.final_cols:
            self.final_cols.remove('temp')
        
        return self
    
    def transform(self, X):
        """Select columns and handle missing values"""
        df = X[self.final_cols].copy()
        
        # ✅ SYNTAX FIX (Avoid FutureWarning)
        df = df.ffill() # Forward fill NaNs
        df = df.bfill() # Backward fill remaining NaNs (at the beginning)
        
        # Final safety net (if the entire column is NaN)
        df = df.fillna(0)
        
        return df


def create_feature_pipeline():
    """
    Creates the complete feature engineering pipeline (Unchanged)
    
    Pipeline steps:
    1. TimeFeatureTransformer - Creates cyclical time features (UPGRADED)
    2. DerivedFeatureTransformer - Creates derived features (UPGRADED)
    3. ColumnPreprocessor - Selects columns and imputes (UPGRADED)
    
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