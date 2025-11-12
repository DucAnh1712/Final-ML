# data_processing.py
import os
import pandas as pd
import config

def create_targets(df, target_col, horizons):
    """Shifts the target column to create future targets."""
    df_new = df.copy()
    for h in horizons:
        df_new[f'target_t{h}'] = df_new[target_col].shift(-h)
    return df_new

def main():
    print("🚀 STARTING DATA PROCESSING (STEP 0)")
    print("="*70)

    # 1. Load Raw Data
    raw_path = os.path.join(config.RAW_DATA_DIR, config.RAW_FILE_NAME)
    try:
        df_raw = pd.read_excel(raw_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Raw data file not found at: {raw_path}")
        return
    except Exception as e:
        print(f"❌ ERROR reading Excel file: {e}")
        return
        
    print(f"✅ Raw file loaded successfully: {raw_path} (Rows: {len(df_raw)})")

    # 2. Basic processing
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_raw = df_raw.sort_values(by='datetime').reset_index(drop=True)

    HORIZONS = [1, 2, 3, 4, 5, 6, 7] # As per new requirement
    
    print(f"Creating targets for T+{HORIZONS}...")
    df_processed = create_targets(df_raw, config.TARGET_COL, HORIZONS)

    # 3. Calculate split points
    n = len(df_processed)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = train_end + int(n * (config.VAL_RATIO))

    print(f"Total rows: {n}")
    print(f"Train split: 0 -> {train_end}")
    print(f"Val split:   {train_end} -> {val_end}")
    print(f"Test split:  {val_end} -> {n}")

    # 4. Split Data
    train_df = df_processed.iloc[:train_end].copy()
    val_df = df_processed.iloc[train_end:val_end].copy()
    test_df = df_processed.iloc[val_end:].copy()

    # 5. Save 3 CSV files
    train_path = os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv")
    val_path = os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv")
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n✅ Saved data_train.csv (Rows: {len(train_df)})")
    print(f"✅ Saved data_val.csv (Rows: {len(val_df)})")
    print(f"✅ Saved data_test.csv (Rows: {len(test_df)})")
    print("\n🎉 DATA PROCESSING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()