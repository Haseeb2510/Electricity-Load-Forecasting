import numpy as np
import pandas as pd            
import xgboost as xgb
from xgboost import XGBRegressor           
import matplotlib.pyplot as plt
import os
import joblib, json
from tensorflow.keras.models import Sequential                                # type: ignore
from pytorch_forecasting import TemporalFusionTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from prepare_data_for_models import splitting_data_eval
from XGBoost_model_training import load_model_xgb, predict_xgb
from LSTM_model_training import load_lstm_model, load_time_steps, predict_lstm, create_eval_sequences
from TFT_model_training import load_tft_model, load_fine_tuned_tft_model, evaluation_TFT


# Root project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_folder = os.path.join(PROJECT_ROOT, 'data')

LSTM_MODEL_FOLDER = os.path.join(PROJECT_ROOT, 'models/LSTM')
TFT_MODEL_FOLDER = os.path.join(PROJECT_ROOT, 'models/TFT')
XGB_MODEL_FOLDER = os.path.join(PROJECT_ROOT, 'models/XGBoost')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

TIME_DATA = 336 # Days of data we are evaluating

def load_data():
    # load test or holdout dataset
    df = pd.read_csv(os.path.join(data_folder, 'worked/engineered_data.csv'), parse_dates=['datetime'])

    # default 336h (14 days), 
    prediction_data, actual_y = splitting_data_eval(df, 360)

    return prediction_data, actual_y

def save_metrics(mae, rmse, mape, report_folder, model_name):
    results = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
    with open(os.path.join(report_folder, f'{model_name}_eval_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

def load_xgb() -> XGBRegressor:
    xgb_model = load_model_xgb()
    return xgb_model

def load_lstm() -> Sequential:
    lstm_model = load_lstm_model()
    return lstm_model

def load_tft() -> TemporalFusionTransformer:
    tft_model = load_tft_model()
    return tft_model

def load_fine_tuned_tft() -> TemporalFusionTransformer:
    fine_tuned_tft_model = load_fine_tuned_tft_model()
    return fine_tuned_tft_model

def model_eval(model_name: str, y_true_for_metrics: np.ndarray, y_pred) -> tuple[float, float, float]:
# Ensure lengths match
    min_length = min(len(y_true_for_metrics), len(y_pred))
    y_true_aligned = y_true_for_metrics[-min_length:]  # Take the most recent values
    y_pred_aligned = y_pred[-min_length:]
    
    # compute metrics
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    rmse = root_mean_squared_error(y_true_aligned, y_pred_aligned)
    mape = np.mean(np.abs((y_true_aligned - y_pred_aligned) / y_true_aligned)) * 100

    print(f"\n📈 Model: {model_name.upper()}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Sample size: {len(y_true_aligned)} predictions")

    # plot results
    plt.figure(figsize=(15,5))
    plt.plot(y_true_aligned[-TIME_DATA:], label="Actual", lw=2)
    plt.plot(y_pred_aligned[-TIME_DATA:], label="Predicted", lw=2, alpha=0.8)
    plt.title(f"{model_name.upper()} Forecast - Last {min_length} hours")
    plt.legend(); plt.grid(True)
    
    # Save plot
    os.makedirs(os.path.join(REPORTS_DIR, model_name), exist_ok=True)
    report_folder = os.path.join(REPORTS_DIR, model_name)
    plt.savefig(os.path.join(report_folder, f"{model_name}_forecast.png"))
    plt.show()
    
    save_metrics(mae, rmse, mape, report_folder, model_name)

    return mae, rmse, mape


def model_selection(prediction_data: pd.DataFrame, actual_y: pd.Series, model_name: str):
    prediction_data = prediction_data.copy()
    model = False
    y_pred = None
    y_true_for_metrics = None  # Store the actual values for metrics

    # model_name: 'xgb', 'lstm', 'tft' or 'tft-tuned'
    try:
        if model_name == 'xgb':
            xgb_model = load_xgb()
            prediction_data = prediction_data.drop(columns=['datetime', 'nat_demand'])
            y_pred = predict_xgb(xgb_model, prediction_data)
            y_true_for_metrics = actual_y.to_numpy()  # Store actual values
            model = True
            
        elif model_name == 'lstm':
            lstm_model = load_lstm()
            time_steps = load_time_steps()
            prediction_data = prediction_data.drop(columns=['datetime', 'nat_demand'])
            print('Time steps:', time_steps)
            y_true_original = actual_y.to_numpy()  # Store original actual values
            y_pred = predict_lstm(lstm_model, prediction_data, time_steps)
            
            # Create sequences but don't overwrite 'actual'
            y_true_sequences = create_eval_sequences(y_true_original, time_steps)
            
            # Align lengths
            min_len = min(len(y_true_sequences), len(y_pred))
            y_true_for_metrics = y_true_sequences[:min_len]
            y_pred = y_pred[:min_len]  # Trim predictions to match
            model = True
            
        elif model_name == 'tft':
            tft_model = load_tft()
            y_pred = evaluation_TFT(tft_model, prediction_data)
            y_true_for_metrics = actual_y.to_numpy()
            model = True
            
        elif model_name == 'tft-tuned':
            tft_tuned_model = load_fine_tuned_tft()
            y_pred = evaluation_TFT(tft_tuned_model, prediction_data)
            y_true_for_metrics = actual_y.to_numpy()
            model = True
            
        else:
            raise ValueError("Invalid model name. Choose from: 'xgb', 'lstm', 'tft', 'tft-tuned'")
    
    except Exception as e:
        print(f"❌ Error loading or predicting with {model_name}: {e}")
        return None, None, None
    
    if model and y_pred is not None and y_true_for_metrics is not None:
        return model_eval(model_name, y_true_for_metrics, y_pred)
    else:
        print(f"❌ No predictions generated for {model_name}")
        return None, None, None

def evaluate_all(prediction_data: pd.DataFrame, actual_y: pd.Series, model_name: str):
    results = {}

    for model_name in ['xgb', 'lstm', 'tft', 'tft-tuned']:
        print("\n" + "="*60)
        print(f"Evaluating: {model_name.upper()}")
        mae, rmse, mape = model_selection(prediction_data, actual_y, model_name)
        results[model_name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    comparison_df = pd.DataFrame(results).T  # rows = models, cols = metrics
    comparison_df = comparison_df.sort_values("MAPE")  # best → worst
    plt.figure(figsize=(10, 5))

    bars = plt.bar(comparison_df.index, comparison_df["MAPE"], color=["#4CAF50", "#FF9800", "#9C27B0", "#03A9F4"])

    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.2,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.ylabel("MAPE (%)", fontsize=12)
    plt.title("Model Comparison by MAPE (Lower is Better)", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    report_folder = os.path.join(REPORTS_DIR, 'model_comparison')
    plt.savefig(os.path.join(report_folder, "model_comparison.png"))
    plt.show()

def main():
    prediction_data, actual_y = load_data()
    model_name = input("Insert the name of model (xgb, lstm, tft, tft-tuned) or enter to evaluate all: ")
    if model_name in ['xgb', 'lstm', 'tft', 'tft-tuned']:
        model_selection(prediction_data, actual_y, model_name)
    else:
        evaluate_all(prediction_data, actual_y, 'model_comparison')

if __name__ == '__main__':
    main()