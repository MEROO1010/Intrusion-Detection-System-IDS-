import pandas as pd
import yaml
import os

def get_project_root():
    """
    Returns the absolute path of the project root directory.
    This works regardless of where the script is executed from.
    """
    # Get the directory of the current file (load_data.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up two levels to reach the project root
    # Example: .../intrusion_detection_system/src/data/ -> .../intrusion_detection_system/
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    return project_root

def load_config():
    """
    Loads the config.yaml file from the project root.
    """
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'config.yaml')

    # Debug: Print the path to verify
    print(f"Looking for config.yaml at: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_raw_data(filepath=None):
    """
    Loads the raw NSL-KDD dataset.
    """
    if filepath is None:
        config = load_config()
        raw_data_path = config['data']['raw']
        filepath = os.path.join(get_project_root(), raw_data_path)

    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    # Debug: Print the filepath to verify
    print(f"Loading dataset from: {filepath}")

    df = pd.read_csv(filepath, header=None, names=column_names)
    return df

def save_processed_data(df, filepath=None):
    """
    Saves the processed DataFrame to a CSV file.
    """
    if filepath is None:
        config = load_config()
        processed_data_path = config['data']['processed']
        filepath = os.path.join(get_project_root(), processed_data_path)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_processed_data(filepath=None):
    """
    Loads the processed dataset.
    """
    if filepath is None:
        config = load_config()
        processed_data_path = config['data']['processed']
        filepath = os.path.join(get_project_root(), processed_data_path)
    return pd.read_csv(filepath)