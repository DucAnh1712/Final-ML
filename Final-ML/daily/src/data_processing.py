# data_processing.py
import os
import pandas as pd
import numpy as np
import yaml
import config # Import from config.py

def load_and_clean_raw(file_path):
    """Tải, dọn dẹp và convert text sang 0/1."""
    print(f"🔍 Loading raw data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Data file not found at: {file_path}")
    
    df = pd.read_excel(file_path)
    df.columns = [col.strip().lower() for col in df.columns]

    if "datetime" not in df.columns:
        raise ValueError("❌ Missing 'datetime' column in dataset!")
        
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # ======================================================
    # === THÊM BƯỚC NÀY (Convert Text sang 0/1) ===
    # ======================================================
    print("🔬 Converting 'conditions' text to 0/1 features...")
    if "conditions" in df.columns:
        df["conditions_lower"] = df["conditions"].astype(str).str.lower()
        df["is_rain"] = df["conditions_lower"].str.contains("rain", na=False).astype(int)
        df["is_cloudy"] = df["conditions_lower"].str.contains("cloud", na=False).astype(int)
        df["is_clear"] = df["conditions_lower"].str.contains("clear", na=False).astype(int)
        # Xóa cột trung gian
        df = df.drop(columns=["conditions_lower"])
    # ======================================================
    
    # Xóa các cột text/metadata không cần thiết
    drop_cols = ["stations", "description", "icon", "name"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    print(f"✅ Raw data cleaned successfully. Shape: {df.shape}")
    return df

def split_by_time(df, train_ratio, val_ratio):
    """Chia data theo thời gian (train, val, test)."""
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"📊 Data split → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

def save_data_summary(output_dir, summary):
    """Lưu tóm tắt dữ liệu ra file YAML."""
    summary_path = os.path.join(output_dir, "data_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, sort_keys=False)
    print(f"🧾 Data summary saved → {summary_path}")

def main():
    """Main pipeline: Load -> Clean -> Split -> Save."""
    raw_file_path = os.path.join(config.RAW_DATA_DIR, config.RAW_FILE_NAME)
    
    df = load_and_clean_raw(raw_file_path)
    train_df, val_df, test_df = split_by_time(df, config.TRAIN_RATIO, config.VAL_RATIO)

    # Lưu các file CSV đã xử lý
    train_path = os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv")
    val_path = os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv")
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"""
✅ Processed data saved:
  ┣━ Train: {train_path}
  ┣━ Val:   {val_path}
  ┗━ Test:  {test_path}
""")
    
    # Lưu tóm tắt
    summary = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "split_ratio": {"train": config.TRAIN_RATIO, "val": config.VAL_RATIO, "test": 1.0 - config.TRAIN_RATIO - config.VAL_RATIO},
        "raw_file": raw_file_path
    }
    save_data_summary(config.PROCESSED_DATA_DIR, summary)

if __name__ == "__main__":
    main()