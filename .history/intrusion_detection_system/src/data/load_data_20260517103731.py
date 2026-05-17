import pandas as pd
import yaml
import os

def get_project_root():
    # Get the absolute path of the project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_config():
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_raw_data(filepath=None):
    if filepath is None:
        config = load_config()
        filepath = os.path.join(get_project_root(), config['data']['raw'])

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
    df = pd.read_csv(filepath, header=None, names=column_names)
    return df

def save_processed_data(df, filepath=None):
    if filepath is None:
        config = load_config()
        filepath = os.path.join(get_project_root(), config['data']['processed'])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_processed_data(filepath=None):
    if filepath is None:
        config = load_config()
        filepath = os.path.join(get_project_root(), config['data']['processed'])
    return pd.read_csv(filepath)