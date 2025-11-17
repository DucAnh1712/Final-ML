# train.py
import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from clearml import Task 
import config
from feature_engineering import create_feature_pipeline

def load_optuna_best_params():
    """Loads the best parameters found during the Optuna search."""
    params_path = os.path.join(config.MODEL_DIR, config.OPTUNA_RESULTS_YAML)
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"❌ {params_path} not found\n"
            f"Please run 'python {config.OPTUNA_SCRIPT_NAME}' first!"
        )
    with open(params_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    print(f"✅ Loaded Optuna best params from: {params_path}")
    return data['best_params']

def align_data_final(X_feat_scaled_df, y_raw_series):
    """Aligns X features with y target and drops rows where target is NaN."""
    y_aligned = y_raw_series.copy()
    y_aligned.index = X_feat_scaled_df.index 
    y_df = pd.DataFrame(y_aligned)
    
    # Combine X and y based on index
    combined = pd.concat([y_df, X_feat_scaled_df], axis=1)
    
    # Drop NaNs, ensuring data is clean after scaling/alignment
    combined_clean = combined.dropna()
    
    y_final = combined_clean[y_aligned.name]
    X_final = combined_clean.drop(columns=[y_aligned.name])
    return X_final, y_final

def create_model_from_params(params):
    """Instantiates the correct linear model with tuned parameters."""
    model_type = params.get('model_type', 'LinearRegression')
    alpha = params.get('alpha', 1.0)
    l1_ratio = params.get('l1_ratio', 0.5)

    if model_type == 'Ridge':
        print(f"   Model: Ridge (alpha={alpha:.4f})")
        return Ridge(alpha=alpha, random_state=42)
    elif model_type == 'Lasso':
        print(f"   Model: Lasso (alpha={alpha:.4f})")
        # Set max_iter higher for convergence robustness
        return Lasso(alpha=alpha, random_state=42, max_iter=2000)
    elif model_type == 'ElasticNet':
        print(f"   Model: ElasticNet (alpha={alpha:.4f}, l1_ratio={l1_ratio:.4f})")
        # Set max_iter higher for convergence robustness
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)
    else:
        print("   Model: LinearRegression (Default)")
        return LinearRegression(n_jobs=-1)

def main():
    """Main function to load params, train models, and save artifacts."""
    task = Task.init(
        project_name=config.CLEARML_PROJECT_NAME,
        task_name=config.CLEARML_TASK_NAME + " (Production)",
        tags=["Production", "LinearTuned", "Multi-Horizon"]
    )
    
    try:
        all_best_params = load_optuna_best_params()
    except FileNotFoundError as e:
        print(str(e))
        return

    print(f"🚀 STARTING PRODUCTION TRAINING (Tuned Linear, Multi-Horizon)")
    print("="*70)

    # ======================================================
    # 1. LOAD DATA (Merge Train + Val)
    # ======================================================
    print(f"📂 Loading data (Train+Val)...")
    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "data_val.csv"))
    all_train_data = pd.concat([train_df, val_df], ignore_index=True)
    all_train_data['datetime'] = pd.to_datetime(all_train_data['datetime'])
    all_train_data = all_train_data.set_index('datetime', drop=False)
    X_train_full = all_train_data.copy()

    # ======================================================
    # 2. FIT PIPELINE & SCALER (ONCE on full training data)
    # ======================================================
    feature_pipeline = create_feature_pipeline()
    scaler = RobustScaler()

    print("Fitting Feature Pipeline on full training data (Train+Val)...")
    X_feat_full = feature_pipeline.fit_transform(X_train_full)
    
    print("Fitting Scaler on full training data...")
    # Clean features before fitting the scaler
    X_feat_full_clean = X_feat_full.dropna() 
    scaler.fit(X_feat_full_clean)
    
    # Save individual components (keep them)
    joblib.dump(feature_pipeline, os.path.join(config.MODEL_DIR, config.PIPELINE_NAME))
    joblib.dump(scaler, os.path.join(config.MODEL_DIR, config.SCALER_NAME))
    print(f"💾 Feature Pipeline saved to: {config.PIPELINE_NAME}")
    print(f"💾 Scaler saved to: {config.SCALER_NAME}")

    # ======================================================
    # 2.5. ADDED: CREATE AND SAVE PIPELINE FOR ONNX
    # ======================================================
    print("\n" + "-"*50)
    print("Creating ONNX-convertible preprocessing pipeline...")
    
    # Combine feature_pipeline AND scaler (both are already fitted)
    onnx_preprocessing_pipeline = Pipeline([
        ('feature_engineering', feature_pipeline),
        ('scaler', scaler)
    ])
    
    # This is the file name that 'convert_to_onnx.py' is looking for
    onnx_pipeline_filename = 'onnx_convertible_pipeline.pkl'
    onnx_pipeline_path = os.path.join(config.MODEL_DIR, onnx_pipeline_filename)
    
    joblib.dump(onnx_preprocessing_pipeline, onnx_pipeline_path)
    
    print(f"✅ Preprocessing pipeline for ONNX (Features + Scaler) saved to:")
    print(f"   {onnx_pipeline_path}")
    print("-"*50 + "\n")
    # ======================================================
    # END OF ADDED SECTION
    # ======================================================


    # ======================================================
    # 3. LOOP AND TRAIN EACH MODEL
    # ======================================================
    all_train_metrics = {}
    
    # Transform all features using the fitted scaler
    X_scaled_full = scaler.transform(X_feat_full)
    X_scaled_full_df = pd.DataFrame(X_scaled_full, index=X_feat_full.index, columns=X_feat_full.columns)

    # ✅ IMPORTANT LOOP: Iterate through each forecast horizon
    for target_name in config.TARGET_FORECAST_COLS: # Will loop 7 times
        print("\n" + "="*30)
        print(f"🎯 Training for: {target_name}")
        print("="*30)
        
        y_train_full = all_train_data[target_name]

        # 4. Align data
        X_final_train, y_final_train = align_data_final(
            X_scaled_full_df, y_train_full
        )
        print(f"📊 Final training data (aligned): X={X_final_train.shape}, y={y_final_train.shape}")
        
        # 5. FIT MODEL (FROM TUNE RESULTS)
        if target_name not in all_best_params:
            print(f"⚠️ Tuned params not found for {target_name}. Using default LinearRegression.")
            model = LinearRegression(n_jobs=-1)
        else:
            best_params = all_best_params[target_name]
            # Log best parameters to ClearML
            task.connect(best_params, name=f'Best Params ({target_name})')
            model = create_model_from_params(best_params)
        
        print(f"⏳ Training final {target_name} model...")
        model.fit(X_final_train, y_final_train)
        print(f"✅ Training complete!")

        # 6. CALCULATE TRAIN METRICS
        y_train_pred = model.predict(X_final_train)
        train_metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(y_final_train, y_train_pred))),
            "MAE": float(mean_absolute_error(y_final_train, y_train_pred)),
            "R2": float(r2_score(y_final_train, y_train_pred))
        }
        all_train_metrics[target_name] = train_metrics
        print(f"   Train RMSE: {train_metrics['RMSE']:.4f}")

        # 7. SAVE MODEL (separate name for each target)
        model_name = f"{target_name}_{config.MODEL_NAME}"
        model_path = os.path.join(config.MODEL_DIR, model_name)
        joblib.dump(model, model_path)
        print(f"💾 Model saved to: {model_path}")

    # Save all train metrics
    metrics_path = os.path.join(config.OUTPUT_DIR, config.TRAIN_METRICS_NAME)
    with open(metrics_path, "w") as f:
        yaml.dump(all_train_metrics, f, sort_keys=False)
    print(f"\n💾 All train metrics saved to: {metrics_path}")
    
    print(f"\n🚀 NEXT STEP: Run 'python inference.py'")
    task.close()

if __name__ == "__main__":
    main()