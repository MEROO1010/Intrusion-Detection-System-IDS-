import os
import pandas as pd
import yaml

def load_config(config_path=None):
    if config_path is None:
        # Resolve config relative to this file's root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def read_raw_data(file_path, config):
    """Loads the raw NSL-KDD dataset and structurally maps headers."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing raw data file target at: {file_path}")
    df = pd.read_csv(file_path, header=None, names=config["data"]["columns"])
    if "difficulty_score" in df.columns:
        df = df.drop(columns=["difficulty_score"])
    return df