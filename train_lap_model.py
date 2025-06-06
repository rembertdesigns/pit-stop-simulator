# train_lap_model.py (Enhanced and Corrected Version)

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration ---

# 1. Define paths to your simulation log data (CSV files)
DATA_LOG_FILES = [
    "logs/gym_race_lap_data.csv", # Ensure this file exists and contains recent data
    # Add more files if you have them, e.g., "logs/run_2_data.csv"
]

# 2. Define the path to save your retrained model
MODEL_SAVE_PATH = "models/lap_time_predictor.pkl"

# 3. Define the list of RAW feature columns to select from your CSV/DataFrame
#    These must match the column names in your log files
RAW_FEATURE_COLUMNS_FROM_LOG = [
    "lap", "tire_wear", "traffic",
    "fuel_weight",
    "track_temperature",
    "grip_factor",
    "rain",
    "safety_car_active", # Use the name as it appears in your CSV
    "vsc_active",
    "tire_type" # This will be one-hot encoded
]

# 4. Define the target variable
TARGET_COLUMN = "lap_time"

# --- Preprocessing Function (Corrected Version) ---
def preprocess_data_for_training(df_list):
    """Loads, combines, and preprocesses data for model training."""
    if not df_list:
        print("No dataframes provided for preprocessing.")
        return pd.DataFrame(), []

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined data shape before processing: {combined_df.shape}")
    
    initial_cols_to_process = [col for col in RAW_FEATURE_COLUMNS_FROM_LOG + [TARGET_COLUMN] if col in combined_df.columns]
    processed_df = combined_df[initial_cols_to_process].copy()

    # 1. Handle missing target values
    if TARGET_COLUMN not in processed_df.columns or processed_df[TARGET_COLUMN].isnull().all():
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing or all NaN.")
    processed_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    if processed_df.empty:
        print("No data remaining after dropping NaN target values. Exiting."); return pd.DataFrame(), []

    # 2. Convert boolean-like columns to 0/1 integers
    bool_cols_to_convert = ['rain', 'safety_car_active', 'vsc_active']
    for col in bool_cols_to_convert:
        if col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                 processed_df[col] = processed_df[col].astype(str).str.lower().map(
                     {'true': 1, 'false': 0, '1':1, '0':0, '1.0':1, '0.0':0}).fillna(0)
            processed_df[col] = processed_df[col].astype(int)
        else:
            print(f"Warning: Boolean-like column '{col}' missing, adding with default 0.")
            processed_df[col] = 0 

    # --- Renaming block removed to maintain consistency with 'safety_car_active' ---

    # 3. One-hot encode 'tire_type'
    if "tire_type" in processed_df.columns:
        try:
            tire_dummies = pd.get_dummies(processed_df["tire_type"], prefix="tire_type", dummy_na=False)
            processed_df = pd.concat([processed_df, tire_dummies], axis=1)
        except Exception as e:
            print(f"Error during one-hot encoding 'tire_type': {e}")
    
    # 4. Define the final list of feature names for the model
    final_model_feature_names = []
    
    # This list must use 'safety_car_active' to match the raw data and other scripts
    base_model_features = [
        "lap", "tire_wear", "traffic", "fuel_weight", 
        "track_temperature", "grip_factor",
        "rain", "safety_car_active", "vsc_active" # <<< CORRECTED
    ]
    for col in base_model_features:
        if col in processed_df.columns:
            final_model_feature_names.append(col)
        else: 
            print(f"Warning: Model feature '{col}' missing, adding with default 0.")
            processed_df[col] = 0 
            final_model_feature_names.append(col)

    # Add one-hot encoded tire type columns
    possible_tire_dummy_names = [f"tire_type_{t}" for t in ["soft", "medium", "hard", "intermediate", "wet"]]
    for col in possible_tire_dummy_names:
        if col in processed_df.columns:
            final_model_feature_names.append(col)
        else:
            processed_df[col] = 0 
            final_model_feature_names.append(col)
            
    # 5. Handle any remaining NaNs in the final selected feature columns
    for col in final_model_feature_names:
        if processed_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(processed_df[col]):
                median_val = processed_df[col].median()
                processed_df[col].fillna(median_val, inplace=True)
    
    print(f"Final features for model training ({len(final_model_feature_names)}): {final_model_feature_names}")
    return processed_df, final_model_feature_names

# --- Main Training Function ---
def train_lap_time_model():
    print("--- Starting Lap Time Predictor Model Training ---")
    all_dataframes = []
    for file_path in DATA_LOG_FILES:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    all_dataframes.append(df)
                    print(f"Successfully loaded and added {file_path}, shape: {df.shape}")
                else: print(f"Warning: Data file {file_path} is empty.")
            except Exception as e: print(f"Error loading or processing {file_path}: {e}")
        else: print(f"Warning: Data file not found - {file_path}")

    if not all_dataframes:
        print("No data could be loaded. Please ensure your DATA_LOG_FILES list is correct and files contain data."); return
        
    processed_df, model_feature_names = preprocess_data_for_training(all_dataframes)
    
    if processed_df.empty or not model_feature_names:
        print("Preprocessing resulted in no data or no features. Exiting training."); return

    X = processed_df[model_feature_names]
    y = processed_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples, Test set size: {X_test.shape[0]} samples")

    if X_train.empty or y_train.empty:
        print("Training data is empty after split. Cannot train model."); return

    print(f"\nTraining RandomForestRegressor model with {len(model_feature_names)} features...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=10, min_samples_leaf=5, oob_score=True)
    model.fit(X_train, y_train)
    print("Model training complete.")
    if hasattr(model, 'oob_score_'): print(f"  Model Out-of-Bag (OOB) R2 Score: {model.oob_score_:.3f}")

    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test); rmse = np.sqrt(mse); r2 = r2_score(y_test, y_pred_test)
    print(f"\n--- Model Evaluation on Test Set ---")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"  R-squared (R2 Score):           {r2:.3f}")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': model_feature_names, 'importance': importances}).sort_values(by='importance', ascending=False).reset_index(drop=True)
        print("\nTop 10 Feature Importances:"); print(feature_importance_df.head(10))

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\n✅ Model retrained and saved to: {MODEL_SAVE_PATH}")
    
    print("\nIMPORTANT: Your `PitStopEnv` and `streamlit_app.py` must now prepare this exact same feature set for prediction.")

if __name__ == '__main__':
    # BEFORE RUNNING:
    # 1. Ensure you have CSV log data (e.g., 'logs/gym_race_lap_data.csv') from your PitStopEnv.
    # 2. Update DATA_LOG_FILES list at the top of this script if needed.
    # 3. Update RAW_FEATURE_COLUMNS_FROM_LOG if your CSV column names differ.
    train_lap_time_model()