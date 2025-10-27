# feature_engineering.py
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import config

# ======================================================
# 1. CUSTOM TRANSFORMER CLASSES (Step 4)
# ======================================================
class FFillImputer(BaseEstimator, TransformerMixin):
    """
    Apply .fillna(method='ffill') and .fillna(method='bfill')
    ONLY to numeric columns.
    This respects time-series order and is safe from data leakage.
    """
    def __init__(self):
        self.num_cols = []

    def fit(self, X, y=None):
        # 1. FIT: Chỉ cần tìm ra cột nào là cột số
        self.num_cols = X.select_dtypes(include=[np.number]).columns
        return self
    
    def transform(self, X):
        # 2. TRANSFORM: Áp dụng fill
        df = X.copy()
        if not self.num_cols.empty:
            # Dùng ffill để điền NaNs bằng giá trị quá khứ
            df[self.num_cols] = df[self.num_cols].ffill().bfill()
        return df
    
class TimeFeatures(BaseEstimator, TransformerMixin):
    """Create time-based features (seasonal, cyclical)."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["month"] = df["datetime"].dt.month
        df["dayofyear"] = df["datetime"].dt.dayofyear
        df["weekofyear"] = df["datetime"].dt.isocalendar().week
        df["dayofweek"] = df["datetime"].dt.dayofweek
        
        # Cyclical features
        df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
        df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
        df["sin_week"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
        df["cos_week"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
        return df

class LagRollingFeatures(BaseEstimator, TransformerMixin):
    """Create lag and rolling statistics features."""
    def __init__(self, lag_cols, lags, windows):
        self.lag_cols = lag_cols
        self.lags = lags
        self.windows = windows
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Important: must sort by datetime to calculate lags/rolling correctly
        df = df.sort_values("datetime").reset_index(drop=True) 
        
        for col in self.lag_cols:
            if col in df.columns:
                for lag in self.lags:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)
                for win in self.windows:
                    # Use shift(1) to ensure only past data is used (no data leakage)
                    rolling_series = df[col].shift(1).rolling(win, min_periods=1)
                    df[f"{col}_rollmean{win}"] = rolling_series.mean()
                    df[f"{col}_rollstd{win}"] = rolling_series.std()
        return df

class TextFeatureTransformer(BaseEstimator, TransformerMixin):
    """Handle text columns (if any) - e.g., conditions."""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        if "conditions" in df.columns:
            df["conditions"] = df["conditions"].astype(str).str.lower()
            df["is_rain"] = df["conditions"].str.contains("rain", na=False).astype(int)
            df["is_cloudy"] = df["conditions"].str.contains("cloud", na=False).astype(int)
            df["is_clear"] = df["conditions"].str.contains("clear", na=False).astype(int)
        return df

class DropTextCols(BaseEstimator, TransformerMixin):
    """Drop text/unnecessary columns."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        # Add 'datetime' to this list
        drop_cols = ["conditions", "preciptype", "sunrise", "sunset", "datetime"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
        return df
    
# ======================================================
# 2. FEATURE PIPELINE CREATION FUNCTION
# ======================================================

def create_feature_pipeline():
    """Create Scikit-learn Pipeline for feature engineering."""
    feature_pipeline = Pipeline([
        ('imputer', FFillImputer()),
        ('time', TimeFeatures()),
        ('weather_text', TextFeatureTransformer()),
        ('lags', LagRollingFeatures(
            lag_cols=config.LAG_COLS, 
            lags=config.LAGS, 
            windows=config.WINDOWS
        )),
        ('drop_text', DropTextCols())
    ])
    return feature_pipeline

# ======================================================
# 3. MAIN PROCESS (FIXED DATA LEAKAGE)
# ======================================================

def main():
    """
    Main process: Load processed data -> Concat -> Create features -> Split -> Save.
    This is the standard way to avoid data leakage at the train/val/test boundaries.
    """
    print("🚀 Starting Feature Engineering process...")
    
    # 1. Load processed data
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    test_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv"))
    
    # Store lengths to split later
    train_len = len(train_df)
    val_len = len(val_df)

    # 2. Concatenate
    # Concat train, val, and test to transform together.
    # This ensures lags for val are calculated from train, and lags for test from val.
    df_full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_full["datetime"] = pd.to_datetime(df_full["datetime"])
    df_full = df_full.sort_values("datetime").reset_index(drop=True)

    # 3. Create pipeline and transform
    feature_pipeline = create_feature_pipeline()
    
    print("⚙️ Applying feature engineering pipeline to full dataset...")
    # Fit on train data only
    feature_pipeline.fit(train_df)
    # Transform the entire dataset
    df_full_feat = feature_pipeline.transform(df_full)
    
    # 4. Split again
    train_feat = df_full_feat.iloc[:train_len]
    val_feat = df_full_feat.iloc[train_len : train_len + val_len]
    test_feat = df_full_feat.iloc[train_len + val_len :]

    # 5. Handle NaNs
    # Only drop NaNs (from lag/rolling) on the train set
    # val and test will not have NaNs at the start (as they use lags from train/val)
    train_feat = train_feat.dropna().reset_index(drop=True)
    val_feat = val_feat.reset_index(drop=True)
    test_feat = test_feat.reset_index(drop=True)

    print(f"📊 Shape after feature creation: Train={train_feat.shape}, Val={val_feat.shape}, Test={test_feat.shape}")

    # 6. Save results
    train_path = os.path.join(config.FEATURE_DIR, "feature_train.csv")
    val_path = os.path.join(config.FEATURE_DIR, "feature_val.csv")
    test_path = os.path.join(config.FEATURE_DIR, "feature_test.csv")

    train_feat.to_csv(train_path, index=False)
    val_feat.to_csv(val_path, index=False)
    test_feat.to_csv(test_path, index=False)
    
    print(f"""
✅ Feature data saved:
  ┣━ Train: {train_path}
  ┣━ Val:   {val_path}
  ┗━ Test:  {test_path}
""")

if __name__ == "__main__":
    main()