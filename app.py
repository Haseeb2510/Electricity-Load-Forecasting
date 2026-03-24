from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
import sys
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import your existing modules
from XGBoost_model_training import load_model_xgb, predict_xgb
from LSTM_model_training import load_lstm_model, load_time_steps, predict_lstm, create_eval_sequences
from TFT_model_training import load_tft_model, load_fine_tuned_tft_model, evaluation_TFT
from prepare_data_for_models import splitting_data_eval
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Root project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(PROJECT_ROOT, 'data')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Model configurations
MODEL_CONFIG = {
    'xgb': {
        'name': 'XGBoost',
        'horizon': 24,
        'needs_sequences': False,
        'description': 'Gradient boosting model - best for short-term forecasting'
    },
    'lstm': {
        'name': 'LSTM',
        'horizon': 360,
        'needs_sequences': True,
        'description': 'Long Short-Term Memory - good for capturing temporal patterns'
    },
    'tft': {
        'name': 'TFT',
        'horizon': 360,
        'needs_sequences': False,
        'description': 'Temporal Fusion Transformer - advanced deep learning for time series'
    },
    'tft-tuned': {
        'name': 'TFT (Fine-tuned)',
        'horizon': 360,
        'needs_sequences': False,
        'description': 'Fine-tuned TFT model for better performance'
    }
}

def load_data_for_model(model_name, horizon):
    """Load data for specific model"""
    try:
        df = pd.read_csv(os.path.join(data_folder, 'worked/engineered_data.csv'), parse_dates=['datetime'])
        report_folder = os.path.join(REPORTS_DIR, model_name)
        os.makedirs(report_folder, exist_ok=True)
        
        prediction_data, actual_y = splitting_data_eval(df, report_folder, model_name, horizon)
        return prediction_data, actual_y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def run_prediction(model_name, prediction_data, actual_y):
    """Run prediction for a specific model"""
    try:
        prediction_data = prediction_data.copy()
        
        if model_name == 'xgb':
            model = load_model_xgb()
            prediction_data = prediction_data.drop(columns=['datetime', 'nat_demand'])
            y_pred = predict_xgb(model, prediction_data)
            y_true = actual_y.to_numpy()
            
        elif model_name == 'lstm':
            model = load_lstm_model()
            time_steps = load_time_steps()
            prediction_data = prediction_data.drop(columns=['datetime', 'nat_demand'])
            y_true_original = actual_y.to_numpy()
            y_pred = predict_lstm(model, prediction_data, time_steps)
            
            y_true_sequences = create_eval_sequences(y_true_original, time_steps)
            min_len = min(len(y_true_sequences), len(y_pred))
            y_true = y_true_sequences[:min_len]
            y_pred = y_pred[:min_len]
            
        elif model_name == 'tft':
            model = load_tft_model()
            y_pred = evaluation_TFT(model, prediction_data)
            y_true = actual_y.to_numpy()
            
        elif model_name == 'tft-tuned':
            model = load_fine_tuned_tft_model()
            y_pred = evaluation_TFT(model, prediction_data)
            y_true = actual_y.to_numpy()
            
        else:
            return None, None, None, "Invalid model name"
        
        # Align predictions and actual values
        min_len = min(len(y_true), len(y_pred))
        y_true_aligned = y_true[-min_len:]
        y_pred_aligned = y_pred[-min_len:]
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
        rmse = root_mean_squared_error(y_true_aligned, y_pred_aligned)
        mape = np.mean(np.abs((y_true_aligned - y_pred_aligned) / y_true_aligned)) * 100
        
        return y_true_aligned, y_pred_aligned, {
            'MAE': f"{mae:.2f}",
            'RMSE': f"{rmse:.2f}",
            'MAPE': f"{mape:.2f}%",
            'samples': len(y_true_aligned)
        }, None
        
    except Exception as e:
        traceback.print_exc()
        return None, None, None, str(e)

def create_plot(y_true, y_pred, model_name, title):
    """Create plot and return as base64 encoded image"""
    plt.figure(figsize=(15, 5))
    
    # Plot last 200 points or all if less
    n_points = min(200, len(y_true))
    
    plt.plot(y_true[-n_points:], label='Actual', lw=2, color='blue')
    plt.plot(y_pred[-n_points:], label='Predicted', lw=2, alpha=0.8, color='red')
    plt.title(f'{title} - Last {n_points} Hours')
    plt.xlabel('Time Steps')
    plt.ylabel('Electricity Demand')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    """Main page with model selection"""
    return render_template('index.html', models=MODEL_CONFIG)

@app.route('/predict', methods=['POST'])
def predict():
    """Run prediction for selected model"""
    model_name = request.form.get('model')
    
    if model_name not in MODEL_CONFIG:
        return jsonify({'error': 'Invalid model selection'}), 400
    
    config = MODEL_CONFIG[model_name]
    horizon = config['horizon']
    
    # Load data
    prediction_data, actual_y = load_data_for_model(model_name, horizon)
    
    if prediction_data is None:
        return jsonify({'error': 'Failed to load data'}), 500
    
    # Run prediction
    y_true, y_pred, metrics, error = run_prediction(model_name, prediction_data, actual_y)
    
    if error:
        return jsonify({'error': error}), 500
    
    # Create plot
    plot_url = create_plot(y_true, y_pred, model_name, config['name'])
    
    # Prepare results
    results = {
        'model_name': config['name'],
        'model_id': model_name,
        'description': config['description'],
        'metrics': metrics,
        'plot': plot_url,
        'predictions': y_pred.tolist()[-50:],  # Last 50 predictions
        'actual': y_true.tolist()[-50:]       # Last 50 actual values
    }
    
    return render_template('index.html', models=MODEL_CONFIG, results=results)

@app.route('/compare', methods=['POST'])
def compare():
    """Compare all models"""
    all_results = {}
    
    for model_name, config in MODEL_CONFIG.items():
        print(f"Evaluating {model_name}...")
        horizon = config['horizon']
        
        # Load data
        prediction_data, actual_y = load_data_for_model(model_name, horizon)
        
        if prediction_data is None:
            all_results[model_name] = {'error': 'Failed to load data'}
            continue
        
        # Run prediction
        y_true, y_pred, metrics, error = run_prediction(model_name, prediction_data, actual_y)
        
        if error:
            all_results[model_name] = {'error': error}
        else:
            all_results[model_name] = {
                'name': config['name'],
                'metrics': metrics,
                'plot': create_plot(y_true, y_pred, model_name, config['name'])
            }
    
    return render_template('index.html', models=MODEL_CONFIG, comparison=all_results)

@app.route('/api/predict/<model_name>', methods=['GET'])
def api_predict(model_name):
    """REST API endpoint for predictions"""
    if model_name not in MODEL_CONFIG:
        return jsonify({'error': 'Invalid model'}), 400
    
    config = MODEL_CONFIG[model_name]
    prediction_data, actual_y = load_data_for_model(model_name, config['horizon'])
    
    if prediction_data is None:
        return jsonify({'error': 'Failed to load data'}), 500
    
    y_true, y_pred, metrics, error = run_prediction(model_name, prediction_data, actual_y)
    
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify({
        'model': model_name,
        'metrics': metrics,
        'predictions': y_pred.tolist()[:100],  # First 100 predictions
        'actual': y_true.tolist()[:100]
    })

if __name__ == '__main__':
    # Create reports directory if it doesn't exist
    os.makedirs(REPORTS_DIR, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)