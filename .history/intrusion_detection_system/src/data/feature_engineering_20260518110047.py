import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.data.load_data import load_config, read_raw_data

def preprocess_pipeline():
    config = load_config()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    
    raw_train_path = os.path.join(base_dir, config["data"]["raw_train_path"])
    raw_test_path = os.path.join(base_dir, config["data"]["raw_test_path"])
    
    train_df = read_raw_data(raw_train_path, config)
    test_df = read_raw_data(raw_test_path, config)
    
    # Binary target configuration: normal = 0, any attack profile = 1
    train_df["target"] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
    test_df["target"] = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)
    
    train_df = train_df.drop(columns=["label"])
    test_df = test_df.drop(columns=["label"])
    
    # One-Hot Encoding categorical structures safely across partitions
    categorical_cols = ["protocol_type", "service", "flag"]
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)
    
    train_processed = combined.iloc[:len(train_df)].copy()
    test_processed = combined.iloc[len(train_df):].copy()
    
    X_train = train_processed.drop(columns=["target"])
    X_test = test_processed.drop(columns=["target"])
    
    # Feature Scaling Scaling state preservation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_scaled_df["target"] = train_processed["target"].values
    
    test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_scaled_df["target"] = test_processed["target"].values
    
    # Write processed assets
    os.makedirs(os.path.join(base_dir, "data/processed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    
    train_scaled_df.to_csv(os.path.join(base_dir, config["data"]["processed_train_path"]), index=False)
    test_scaled_df.to_csv(os.path.join(base_dir, config["data"]["processed_test_path"]), index=False)
    
    joblib.dump(scaler, os.path.join(base_dir, config["models"]["scaler_path"]))
    print("✓ Data pipeline preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_pipeline()