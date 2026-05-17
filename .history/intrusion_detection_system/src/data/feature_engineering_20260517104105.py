import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import yaml
import os

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

def encode_categorical(df):
    categorical_cols = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def normalize_features(df):
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

def handle_class_imbalance(X, y):
    config = load_config()
    smote = SMOTE(random_state=config['smote']['random_state'])
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def preprocess_data(df):
    df = df.drop(['difficulty'], axis=1, errors='ignore')
    df, label_encoders = encode_categorical(df)
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    return df, label_encoders