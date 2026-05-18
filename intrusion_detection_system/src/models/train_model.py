import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.data.load_data import load_config
from src.models.model_utils import save_model, evaluate_predictions

def train():
    config = load_config()
    
    # Load processed data
    train_df = pd.read_csv(config["data"]["processed_train_path"])
    test_df = pd.read_csv(config["data"]["processed_test_path"])
    
    X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
    X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]
    
    # 1. Train Random Forest
    rf_cfg = config["hyperparameters"]["random_forest"]
    rf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"], 
        max_depth=rf_cfg["max_depth"], 
        random_state=rf_cfg["random_state"]
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    evaluate_predictions(y_test, rf_preds, "Random Forest")
    save_model(rf, config["models"]["rf_path"])
    
    # 2. Train XGBoost
    xgb_cfg = config["hyperparameters"]["xgboost"]
    xgb = XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        random_state=xgb_cfg["random_state"]
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    evaluate_predictions(y_test, xgb_preds, "XGBoost")
    save_model(xgb, config["models"]["xgb_path"])

if __name__ == "__main__":
    train()