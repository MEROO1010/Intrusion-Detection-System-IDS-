import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

# Append project root to path to ensure correct src routing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.load_data import load_config
from src.models.model_utils import load_model

app = Flask(__name__)

config = load_config()
model = load_model(config["models"]["xgb_path"])  # Defaulting to XGBoost
scaler = load_model(config["models"]["scaler_path"])

# Retrieve feature schema expectations minus target
train_sample = pd.read_csv(config["data"]["processed_train_path"], nrows=1).drop(columns=["target"])
FEATURE_COLUMNS = train_sample.columns.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON payloads representing raw inputs
        data = request.get_json(force=True)
        
        # Build base template matching training features set to zero
        input_data = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS)
        
        # Populate incoming values and handle one-hot encoding dynamically
        for key, value in data.items():
            if key in ["protocol_type", "service", "flag"]:
                # Construct expected dummy column format (e.g., protocol_type_tcp)
                dummy_col = f"{key}_{value}"
                if dummy_col in input_data.columns:
                    input_data.at[0, dummy_col] = 1
            elif key in input_data.columns:
                input_data.at[0, key] = float(value)
        
        # Scale inputs using saved pipeline configurations
        scaled_input = scaler.transform(input_data)
        
        # Inference execution
        prediction = int(model.predict(scaled_input)[0])
        probability = float(model.predict_proba(scaled_input)[0][prediction])
        
        result = "Anomaly Detected" if prediction == 1 else "Normal Traffic"
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'confidence': f"{probability * 100:.2f}%"
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)