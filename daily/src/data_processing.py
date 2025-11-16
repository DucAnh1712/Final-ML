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

    # 3. Create target columns for multiple horizons
    HORIZONS = config.FORECAST_HORIZONS    
    print(f"Creating targets for T+{HORIZONS}...")
    df_processed = create_targets(df_raw, config.TARGET_COL, HORIZONS)

    # 3. Calculate split points
    n = len(df_processed)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = train_end + int(n * (config.VAL_RATIO))

    print(f"\nTotal rows: {n}")
    print(f"Train split: 0 -> {train_end} ({config.TRAIN_RATIO*100:.0f}%)")
    print(f"Val split:   {train_end} -> {val_end} ({config.VAL_RATIO*100:.0f}%)")
    print(f"Test split:  {val_end} -> {n} ({(1-config.TRAIN_RATIO-config.VAL_RATIO)*100:.0f}%)")

    # 5. Split Data
    train_df = df_processed.iloc[:train_end].copy()
    val_df = df_processed.iloc[train_end:val_end].copy()
    test_df = df_processed.iloc[val_end:].copy()

    # Print date ranges
    print(f"\n📅 Date Ranges:")
    print(f"  Train: {train_df.index.min()} → {train_df.index.max()}")
    print(f"  Val:   {val_df.index.min()} → {val_df.index.max()}")
    print(f"  Test:  {test_df.index.min()} → {test_df.index.max()}")

    # 6. Save 3 CSV files (with datetime index)
    train_path = os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv")
    val_path = os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv")
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "data_test.csv")

    # ✅ Save with index=True to preserve datetime
    train_df.to_csv(train_path, index=True)
    val_df.to_csv(val_path, index=True)
    test_df.to_csv(test_path, index=True)

    print(f"\n✅ Saved data_train.csv (Rows: {len(train_df)})")
    print(f"✅ Saved data_val.csv (Rows: {len(val_df)})")
    print(f"✅ Saved data_test.csv (Rows: {len(test_df)})")
    print("\n🎉 DATA PROCESSING COMPLETE!")
    print("="*70)
    print("📌 NEXT STEP: Run 'python optuna_search_linear.py'")

if __name__ == "__main__":
    main()