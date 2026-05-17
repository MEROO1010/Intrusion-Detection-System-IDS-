import pandas as pd
import pickle
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.svm import SVC
from src.data.feature_engineering import handle_class_imbalance

def get_project_root():
    """Returns the absolute path of the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return project_root

def load_config():
    """Loads the config.yaml file from the project root."""
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_random_forest(X_train, y_train):
    config = load_config()
    model = RandomForestClassifier(
        n_estimators=config['random_forest']['n_estimators'],
        random_state=config['random_forest']['random_state']
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    config = load_config()
    model = XGBClassifier(
        use_label_encoder=config['xgboost']['use_label_encoder'],
        eval_metric=config['xgboost']['eval_metric'],
        random_state=config['xgboost']['random_state']
    )
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    config = load_config()
    model = SVC(
        kernel=config['svm']['kernel'],
        probability=config['svm']['probability'],
        random_state=config['svm']['random_state']
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm
    }

def save_model(model, model_name):
    config = load_config()
    save_dir = os.path.join(get_project_root(), config['models']['save_dir'])
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, config['models'][model_name])
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def train_and_save_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = handle_class_imbalance(X_train, y_train)

    models = {
        "random_forest": train_random_forest(X_train, y_train),
        "xgboost": train_xgboost(X_train, y_train),
        "svm": train_svm(X_train, y_train)
    }

    results = {}
    for name, model in models.items():
        evaluation = evaluate_model(model, X_test, y_test)
        results[name] = evaluation
        save_model(model, name)

    return results