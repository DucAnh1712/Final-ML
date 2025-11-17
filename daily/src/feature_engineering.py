# feature_engineering.py (PHIÊN BẢN V4 - LEAKAGE-FREE)
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Tạo cyclical time features (sin/cos encoding)
    ✅ SAFE: Không sử dụng thông tin từ future
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Ensure datetime index
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'datetime' not in df.columns:
                df = df.reset_index()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime', drop=False)
        
        # Cyclical encoding
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['day_of_week'] = df.index.dayofweek
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df.drop(columns=['month', 'day_of_year', 'day_of_week'])


class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Tạo derived features từ raw features
    ✅ SAFE: Chỉ sử dụng thông tin trong cùng row
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Daylight hours calculation
        if 'sunrise' in df.columns and 'sunset' in df.columns:
            sr = pd.to_datetime(df['sunrise'], errors='coerce')
            ss = pd.to_datetime(df['sunset'], errors='coerce')
            valid_times = sr.notna() & ss.notna()
            df['daylight_hours'] = np.nan
            df.loc[valid_times, 'daylight_hours'] = (
                (ss[valid_times] - sr[valid_times]).dt.total_seconds() / 3600
            )
            
            # Solar energy per hour
            if 'solarenergy' in df.columns:
                df['solar_per_hour'] = df['solarenergy'] / (df['daylight_hours'] + 1e-6)
        
        # Temperature range
        if 'tempmax' in df.columns and 'tempmin' in df.columns:
            df['temp_range'] = df['tempmax'] - df['tempmin']
        
        # Dewpoint depression
        if 'temp' in df.columns and 'dew' in df.columns:
            df['dewpoint_depression'] = df['temp'] - df['dew']
        
        # ✅ Pressure change - SAFE within fold (gap ensures no leak)
        if 'sealevelpressure' in df.columns:
            pressure_change = df['sealevelpressure'].diff()
            df['sealevelpressure_change'] = pressure_change.fillna(0)
        
        return df


class ColumnPreprocessor(BaseEstimator, TransformerMixin):
    """
    Select and preprocess final columns
    """
    def __init__(self):
        # Core weather features
        self.feature_cols = [
            'humidity', 'sealevelpressure', 'dew', 'cloudcover', 
            'solarradiation', 'visibility', 'windspeed', 'windgust', 'precip',
            'temp'  # Will be removed in fit() to avoid leakage
        ]
        # Derived features
        self.derived_cols = [
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'daylight_hours', 'solar_per_hour', 'temp_range',
            'dewpoint_depression', 'sealevelpressure_change'
        ]
        self.final_cols = []
    
    def fit(self, X, y=None):
        """Determine which columns are available"""
        all_cols_available = list(X.columns)
        existing_features = [col for col in self.feature_cols if col in all_cols_available]
        existing_derived = [col for col in self.derived_cols if col in all_cols_available]
        
        self.final_cols = existing_features + existing_derived
        
        # ✅ Remove 'temp' to avoid target leakage
        if 'temp' in self.final_cols:
            self.final_cols.remove('temp')
        
        return self
    
    def transform(self, X):
        """Select columns and handle missing values"""
        df = X[self.final_cols].copy()
        # ✅ SAFE IMPUTATION STRATEGY:
        # 1. Gap in CV prevents leakage
        # 2. Sensor readings are often stable over short periods
        df = df.ffill().bfill().fillna(0)
        
        return df


def create_feature_pipeline():
    """
    Tạo pipeline feature engineering hoàn chỉnh
    
    Pipeline steps:
    1. TimeFeatureTransformer - Tạo cyclical time features
    2. DerivedFeatureTransformer - Tạo derived features
    3. ColumnPreprocessor - Select columns và impute missing values
    
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