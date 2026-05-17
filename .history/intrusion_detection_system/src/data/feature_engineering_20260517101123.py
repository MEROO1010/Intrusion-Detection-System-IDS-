import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import yaml
import os

def load_config():
    with open('../../config.yaml', 'r') as f:
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
    # Drop difficulty column
    df = df.drop(['difficulty'], axis=1, errors='ignore')

    # Encode categorical variables
    df, label_encoders = encode_categorical(df)

    # Encode labels (0 for normal, 1 for attack)
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    return df, label_encoders