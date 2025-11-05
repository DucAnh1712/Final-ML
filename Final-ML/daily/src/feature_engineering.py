# feature_engineering.py
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import config

# ======================================================
# 1. CUSTOM TRANSFORMER CLASSES (ĐÃ SỬA)
# ======================================================
class FFillImputer(BaseEstimator, TransformerMixin):
    """Điền NaN bằng ffill/bfill, an toàn cho time-series."""
    def __init__(self):
        self.num_cols = []

    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=[np.number]).columns
        return self
    
    def transform(self, X):
        df = X.copy()
        if not self.num_cols.empty:
            df[self.num_cols] = df[self.num_cols].ffill().bfill()
        return df
    
class TimeFeatures(BaseEstimator, TransformerMixin):
    """Tạo features thời gian (cyclical)."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["month"] = df["datetime"].dt.month
        df["dayofyear"] = df["datetime"].dt.dayofyear
        df["weekofyear"] = df["datetime"].dt.isocalendar().week
        df["dayofweek"] = df["datetime"].dt.dayofweek
        
        df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
        df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
        df["sin_week"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
        df["cos_week"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
        return df

class LagRollingFeatures(BaseEstimator, TransformerMixin):
    """
    Tối ưu hóa: Chỉ tạo Rolling (Xu hướng) và dùng pd.concat
    để tránh bị "fragmented" (phân mảnh).
    """
    def __init__(self, lag_cols, lags, windows):
        self.lag_cols = lag_cols
        self.lags = lags 
        self.windows = windows
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = df.sort_values("datetime").reset_index(drop=True) 
        
        features_list = []

        for col in self.lag_cols:
            if col in df.columns:
                
                # (Vòng lặp LAGS đã bị tắt trong config.py)
                for lag in self.lags:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)
                
                # Chỉ chạy vòng lặp Rolling (Xu hướng)
                for win in self.windows:
                    rolling_series = df[col].shift(1).rolling(win, min_periods=1)
                    
                    roll_mean = rolling_series.mean()
                    roll_mean.name = f"{col}_rollmean{win}"
                    
                    roll_std = rolling_series.std()
                    roll_std.name = f"{col}_rollstd{win}"
                    
                    roll_max = rolling_series.max()
                    roll_max.name = f"{col}_rollmax{win}"
                    
                    roll_min = rolling_series.min()
                    roll_min.name = f"{col}_rollmin{win}"
                    
                    features_list.extend([roll_mean, roll_std, roll_max, roll_min])

                    if col in ['precip', 'solarradiation', 'solarenergy', 'snowdepth', 'is_rain']:
                        roll_sum = rolling_series.sum()
                        roll_sum.name = f"{col}_rollsum{win}"
                        features_list.append(roll_sum)
        
        features_df = pd.concat(features_list, axis=1)
        df = pd.concat([df, features_df], axis=1)
        
        return df

# === XÓA CLASS TextFeatureTransformer (vì nó vi phạm luật "chỉ dùng xu hướng") ===

class DropRawFeatures(BaseEstimator, TransformerMixin):
    """
    XÓA TẤT CẢ các cột thô (raw) VÀ CÁC CỘT TARGET
    để tránh data leakage.
    """
    def __init__(self):
        # Liệt kê TẤT CẢ các cột thô
        self.raw_cols_to_drop = [
            # Biến mục tiêu gốc
            "temp", 
            
            # 28 Cột thô
            "tempmax", "tempmin", "feelslikemax", "feelslikemin", "feelslike",
            "dew", "humidity", "precip", "precipprob", "precipcover",
            "preciptype", "snow", "snowdepth", "windgust", "windspeed", 
            "winddir", "sealevelpressure", "cloudcover", "visibility", 
            "solarradiation", "solarenergy", "uvindex", "severerisk", 
            "sunrise", "sunset", "moonphase", "conditions", "datetime",
            
            # Cột 0/1 (cũng là raw data của Ngày T, PHẢI XÓA)
            "is_rain", "is_cloudy", "is_clear",
            
            # Cột thừa (nếu có)
            "stations", "description", "icon", "name",
            
            # === THÊM CÁC CỘT TARGET MỚI VÀO ĐÂY ĐỂ XÓA ===
            "target_t1", "target_t3", "target_t5", "target_t7"
        ]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        cols_to_drop = list(set([col for col in self.raw_cols_to_drop if col in df.columns]))
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        return df
    
# ======================================================
# 2. FEATURE PIPELINE CREATION FUNCTION (Đã sửa)
# ======================================================

def create_feature_pipeline():
    """Tạo Pipeline cho feature engineering."""
    feature_pipeline = Pipeline([
        ('imputer', FFillImputer()),
        ('time', TimeFeatures()),
        
        # ('weather_text', TextFeatureTransformer()), # <-- ĐÃ XÓA
        
        ('lags_rolling', LagRollingFeatures( 
            lag_cols=config.LAG_COLS, 
            lags=config.LAGS, 
            windows=config.WINDOWS
        )),
        ('drop_raw', DropRawFeatures()) # Bước dọn dẹp cuối cùng
    ])
    return feature_pipeline

# ======================================================
# 3. MAIN PROCESS (Giữ nguyên)
# ======================================================
def main():
    """Chạy quy trình: Concat -> Fit -> Transform -> Split -> Save."""
    print("🚀 Starting Feature Engineering process...")
    
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    test_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv"))
    
    train_len = len(train_df)
    val_len = len(val_df)

    df_full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_full["datetime"] = pd.to_datetime(df_full["datetime"])
    df_full = df_full.sort_values("datetime").reset_index(drop=True)

    feature_pipeline = create_feature_pipeline()
    
    print("⚙️ Applying feature engineering pipeline to full dataset...")
    feature_pipeline.fit(train_df)
    df_full_feat = feature_pipeline.transform(df_full)
    
    train_feat = df_full_feat.iloc[:train_len]
    val_feat = df_full_feat.iloc[train_len : train_len + val_len]
    test_feat = df_full_feat.iloc[train_len + val_len :]

    train_feat = train_feat.dropna().reset_index(drop=True)
    val_feat = val_feat.reset_index(drop=True)
    test_feat = test_feat.reset_index(drop=True)

    print(f"📊 Shape after feature creation: Train={train_feat.shape}, Val={val_feat.shape}, Test={test_feat.shape}")

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