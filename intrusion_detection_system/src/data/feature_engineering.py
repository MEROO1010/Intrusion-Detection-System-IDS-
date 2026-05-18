import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.data.load_data import load_config, read_raw_data

def preprocess_data(train_path, test_path, config):
    """Cleans, encodes, and scales the train and test sets consistently."""
    train_df = read_raw_data(train_path, config)
    test_df = read_raw_data(test_path, config)
    
    # Binary classification target mapping
    train_df["target"] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
    test_df["target"] = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)
    
    train_df = train_df.drop(columns=["label"])
    test_df = test_df.drop(columns=["label"])
    
    # Categorical processing
    categorical_cols = ["protocol_type", "service", "flag"]
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)
    
    train_processed = combined.iloc[:len(train_df)].copy()
    test_processed = combined.iloc[len(train_df):].copy()
    
    X_train = train_processed.drop(columns=["target"])
    X_test = test_processed.drop(columns=["target"])
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_scaled_df["target"] = train_processed["target"].values
    
    test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_scaled_df["target"] = test_processed["target"].values
    
    # --- ABSOLUTE PATH LOGIC FOR ARTIFACTS ---
    # Find the root folder relative to this file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    
    # Build complete destination paths
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    # Make sure both physical directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save datasets using explicit paths
    train_scaled_df.to_csv(os.path.join(processed_dir, "train_processed.csv"), index=False)
    test_scaled_df.to_csv(os.path.join(processed_dir, "test_processed.csv"), index=False)
    
    # Save the scaler using explicit absolute pathing
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler successfully saved to absolute path: {scaler_path}")
    
    return train_scaled_df, test_scaled_df

if __name__ == "__main__":
    cfg = load_config()
    # If run directly via command line, resolve relative local testing paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    t_raw = os.path.join(base_dir, cfg["data"]["raw_train_path"])
    v_raw = os.path.join(base_dir, cfg["data"]["raw_test_path"])
    preprocess_data(t_raw, v_raw, cfg)