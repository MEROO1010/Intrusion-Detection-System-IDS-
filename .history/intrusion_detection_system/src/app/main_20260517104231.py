from flask import Flask, request, render_template
import pickle
import os
import yaml
import numpy as np

app = Flask(__name__)

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

def load_model_and_scaler():
    config = load_config()
    model_path = os.path.join(get_project_root(), config['models']['save_dir'], config['models']['xgboost'])
    scaler_path = os.path.join(get_project_root(), config['models']['save_dir'], config['models']['scaler'])

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model_and_scaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = [
        float(data['duration']), int(data['protocol_type']), int(data['service']),
        int(data['flag']), int(data['src_bytes']), int(data['dst_bytes']),
        int(data['land']), int(data['wrong_fragment']), int(data['urgent']),
        int(data['hot']), int(data['num_failed_logins']), int(data['logged_in']),
        int(data['num_compromised']), int(data['root_shell']), int(data['su_attempted']),
        int(data['num_root']), int(data['num_file_creations']), int(data['num_shells']),
        int(data['num_access_files']), int(data['num_outbound_cmds']),
        int(data['is_host_login']), int(data['is_guest_login']), int(data['count']),
        int(data['srv_count']), float(data['serror_rate']), float(data['srv_serror_rate']),
        float(data['rerror_rate']), float(data['srv_rerror_rate']), float(data['same_srv_rate']),
        float(data['diff_srv_rate']), float(data['srv_diff_host_rate']), int(data['dst_host_count']),
        int(data['dst_host_srv_count']), float(data['dst_host_same_srv_rate']),
        float(data['dst_host_diff_srv_rate']), float(data['dst_host_same_src_port_rate']),
        float(data['dst_host_srv_diff_host_rate']), float(data['dst_host_serror_rate']),
        float(data['dst_host_srv_serror_rate']), float(data['dst_host_rerror_rate']),
        float(data['dst_host_srv_rerror_rate'])
    ]

    features = scaler.transform([features])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    if prediction == 0:
        result = "Normal Traffic"
    else:
        result = "Attack Detected"

    return render_template('index.html',
                           prediction_text=f'Prediction: {result}',
                           probability=f'Confidence: {max(probability) * 100:.2f}%')

if __name__ == "__main__":
    app.run(debug=True)