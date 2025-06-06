import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # A good general-purpose regressor
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler # Optional: if you want to scale features

# --- Configuration ---

# 1. Define paths to your simulation log data (CSV files)
#    Your PitStopEnv saves to 'logs/gym_race_lap_data.csv'.
#    If you run multiple simulations, they might overwrite this file unless you rename them.
#    For now, let's assume you have one primary log file, or you can consolidate them.
#    It's best to have data from many diverse simulation runs.
DATA_LOG_FILES = [
    "logs/gym_race_lap_data.csv",
    # "logs/gym_race_lap_data_run2.csv", # Add more if you have them
]

# 2. Define the path to save your retrained model
MODEL_SAVE_PATH = "models/lap_time_predictor.pkl"

# 3. Define the list of raw feature columns you want to select from your CSV/DataFrame
#    These are columns directly available or easily derived from your PitStopEnv's lap_log.
#    Crucially, include 'vsc_active' and other new relevant features.
RAW_FEATURE_COLUMNS_FROM_LOG = [
    "lap", "tire_wear", "traffic",
    "fuel_weight", "track_temperature", "grip_factor",
    "rain", "safety_car_active", "vsc_active", # These should be boolean or 0/1 in your log
    "tire_type" # This will be one-hot encoded
]

# 4. Define the target variable
TARGET_COLUMN = "lap_time"

# --- Preprocessing Function ---
def preprocess_data_for_training(df_list):
    """Loads, combines, and preprocesses data for model training."""
    if not df_list:
        print("No dataframes provided for preprocessing.")
        return pd.DataFrame(), []

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined data shape before any processing: {combined_df.shape}")
    
    # Select only the raw feature columns + target to start with
    cols_to_keep = [col for col in RAW_FEATURE_COLUMNS_FROM_LOG + [TARGET_COLUMN] if col in combined_df.columns]
    processed_df = combined_df[cols_to_keep].copy()

    # 1. Handle missing target values (lap_time)
    if TARGET_COLUMN not in processed_df.columns or processed_df[TARGET_COLUMN].isnull().all():
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing or all NaN.")
    processed_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    if processed_df.empty:
        print("No data remaining after dropping NaN target values.")
        return pd.DataFrame(), []

    # 2. Convert boolean-like columns to 0/1 integers
    #    (rain, safety_car_active, vsc_active)
    bool_cols_to_convert = ['rain', 'safety_car_active', 'vsc_active']
    for col in bool_cols_to_convert:
        if col in processed_df.columns:
            # Handle True/False strings if they exist from CSV read
            if processed_df[col].dtype == 'object':
                 processed_df[col] = processed_df[col].astype(str).str.lower().map({'true': 1, 'false': 0, '1':1, '0':0}).fillna(0)
            processed_df[col] = processed_df[col].astype(int)
        else:
            processed_df[col] = 0 # Add if missing, default to 0 (no event)

    # 3. One-hot encode 'tire_type'
    if "tire_type" in processed_df.columns:
        tire_dummies = pd.get_dummies(processed_df[TARGET_COLUMN], prefix="tire_type", dummy_na=False) # Changed from processed_df[TARGET_COLUMN] to processed_df["tire_type"]
        processed_df = pd.concat([processed_df, tire_dummies], axis=1)
    
    # Define the final list of feature names for the model
    # This includes original numeric/boolean features and the new one-hot encoded tire_type columns.
    final_model_feature_names = []
    base_features_for_model = [
        "lap", "tire_wear", "traffic", "fuel_weight", 
        "track_temperature", "grip_factor",
        "rain", "safety_car_active", "vsc_active" # Ensure these match names after potential renames
    ]
    for col in base_features_for_model:
        if col in processed_df.columns:
            final_model_feature_names.append(col)
        else: # If a base feature is somehow missing, add it with a default (e.g., 0 or median)
            print(f"Warning: Base feature '{col}' missing in data, adding with default 0.")
            processed_df[col] = 0 
            final_model_feature_names.append(col)

    # Add one-hot encoded tire type columns that might have been generated
    possible_tire_dummies = [f"tire_type_{t}" for t in ["soft", "medium", "hard", "intermediate", "wet"]]
    for col in possible_tire_dummies:
        if col in processed_df.columns:
            final_model_feature_names.append(col)
        else: # Ensure all potential tire dummy columns exist for consistent feature set
            processed_df[col] = 0
            final_model_feature_names.append(col)
            
    # 4. Handle any remaining NaNs in feature columns (e.g., with median for numeric)
    for col in final_model_feature_names:
        if processed_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(processed_df[col]):
                median_val = processed_df[col].median()
                processed_df[col].fillna(median_val, inplace=True)
                print(f"Filled NaNs in '{col}' with median: {median_val}")
            else: # For non-numeric, fill with mode or a placeholder
                mode_val = processed_df[col].mode()
                fallback_val = "unknown" if mode_val.empty else mode_val[0]
                processed_df[col].fillna(fallback_val, inplace=True)
                print(f"Filled NaNs in '{col}' with mode/fallback: {fallback_val}")
    
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Final features for model training: {final_model_feature_names}")
    return processed_df, final_model_feature_names

# --- Main Training Function ---
def train_lap_time_model():
    print("--- Starting Lap Time Predictor Model Training ---")

    # 1. Load Data
    all_dataframes = []
    if not DATA_LOG_FILES or not any(os.path.exists(f) for f in DATA_LOG_FILES):
        print(f"ERROR: No data log files found or specified in DATA_LOG_FILES: {DATA_LOG_FILES}")
        print("Please ensure 'logs/gym_race_lap_data.csv' (or other specified files) exist and contain simulation data.")
        print("You may need to run your simulator (e.g., PitStopEnv directly or via Streamlit app) to generate these logs.")
        return

    for file_path in DATA_LOG_FILES:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                all_dataframes.append(df)
                print(f"Successfully loaded {file_path}, shape: {df.shape}")
            except Exception as e:
                print(f"Error loading or processing {file_path}: {e}")
        else:
            print(f"Warning: Data file not found - {file_path}")

    if not all_dataframes:
        print("No data could be loaded. Exiting training.")
        return
        
    # 2. Preprocess Data
    # This function now returns the processed DataFrame and the list of feature names used.
    processed_df, model_feature_names = preprocess_data_for_training(all_dataframes)
    
    if processed_df.empty or not model_feature_names:
        print("Preprocessing resulted in no data or no features. Exiting training.")
        return

    X = processed_df[model_feature_names]
    y = processed_df[TARGET_COLUMN]

    # 3. Split Data into Training and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples, Test set size: {X_test.shape[0]} samples")

    # Optional: Feature Scaling (can be beneficial for some models, less critical for Random Forest)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # 4. Initialize and Train Model (Example: RandomForestRegressor)
    # You can experiment with different models and hyperparameters.
    print(f"\nTraining RandomForestRegressor model with {len(model_feature_names)} features...")
    # model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, 
    #                               max_depth=20, min_samples_split=10, min_samples_leaf=5)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1,
                                  max_depth=25, min_samples_split=5, min_samples_leaf=2,
                                  max_features='sqrt') # Some common hyperparams
    
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate Model on Test Set
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n--- Model Evaluation on Test Set ---")
    print(f"  Mean Squared Error (MSE):      {mse:.3f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"  R-squared (R2 Score):          {r2:.3f}")

    # Display Feature Importances (for tree-based models like RandomForest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': model_feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        print("\nTop 10 Feature Importances:")
        print(feature_importance_df.head(10))

    # 6. Save the Trained Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\n✅ Model retrained and saved to: {MODEL_SAVE_PATH}")
    
    # If you used a scaler, save it too:
    # scaler_save_path = "models/lap_time_scaler.pkl"
    # joblib.dump(scaler, scaler_save_path)
    # print(f"Scaler saved to: {scaler_save_path}")

    print("\nIMPORTANT: Ensure the feature preparation in your `streamlit_app.py` (for the 'ML Insights' section)")
    print("uses the exact same feature set and preprocessing steps as this training script.")
    print(f"The model was trained with these features: {model_feature_names}")

if __name__ == '__main__':
    # Before running, make sure you have CSV log data from your simulator!
    # For example, run your PitStopEnv directly or via Streamlit and ensure 'logs/gym_race_lap_data.csv' is populated.
    # You might want to collect data from multiple diverse simulation runs.
    train_lap_time_model()