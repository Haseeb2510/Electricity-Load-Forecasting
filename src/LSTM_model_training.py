import numpy as np
import pandas as pd            
from sklearn.metrics import mean_absolute_error, root_mean_squared_error  
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model                      # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization    # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau         # type: ignore
from tensorflow.keras.regularizers import l2                                    # type: ignore
import matplotlib.pyplot as plt
import os
from feature_engineering import feature_engineering
from prepare_data_for_models import splitting_data, prepare_data_lstm, get_cols_to_scale
import joblib, json
import shap


# Root project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
MODEL_FOLDER = os.path.join(PROJECT_ROOT, 'models/LSTM')

def save_scaler_info(scaler, scaling_info):
    """Save scaler and scaling info together"""
    scaler_data = {
        'scaler': scaler,
        'scaling_info': scaling_info
    }
    joblib.dump(scaler_data, f'{MODEL_FOLDER}/scaler.joblib')
    print('✅ Scaler and scaling info saved successfully.')

def load_scaler_info():
    """Load scaler and scaling info"""
    try:
        scaler_data = joblib.load(f'{MODEL_FOLDER}/scaler.joblib')
        scaler = scaler_data['scaler']
        scaler_info = scaler_data['scaling_info']
        print('✅ Scaler loaded successfully.')
    except Exception as e:
        print('Error loading scaler info:', e)
        raise

    # Attach scaling info to scaler for easy access
    return scaler, scaler_info

def save(model: Sequential, history: dict, results: dict, time_steps: dict):
    model.save(os.path.join(MODEL_FOLDER, 'lstm_model.h5'))

    joblib.dump(history, os.path.join(MODEL_FOLDER, 'history.joblib'))

    with open(os.path.join(MODEL_FOLDER, 'LSTM_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(MODEL_FOLDER, 'time_steps.json'), 'w') as f:
        json.dump(time_steps, f, indent=4)

    print('✅ Model, history, time steps and metrics saved successfully.')

def load_time_steps() -> int:
    try:
        with open(os.path.join(MODEL_FOLDER, 'time_steps.json'), 'r') as f:
            time_steps_data = json.load(f)
        time_steps = time_steps_data.get('time_steps')
        return time_steps
    except Exception as e:
        print(f'Error: {e}')
        return 0

def load_lstm_model(model_name: str='lstm_model.h5') -> Sequential | Exception:
    try:
        # Load without compiling, then compile manually
        model = load_model(os.path.join(MODEL_FOLDER, model_name), compile=False)
        
        # Recompile with appropriate metrics
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']  # Adjust based on your training
        )
        return model
    
    except Exception as e:
        print('❌ Error loading LSTM model: ', e)
        # Re-raise the exception instead of returning the error object
        raise e

def drop_cols(df: pd.DataFrame):
    df = df.copy()
    cols_to_drop = [c for c in df.columns if c.startswith(('diff_'))]
    df = df.drop(columns=cols_to_drop)
    return df

def scale_data(X: np.ndarray) -> np.ndarray:
    """
    Scale data using saved scaler and feature indices
    Works directly with numpy arrays (same format used during training)
    """
    try:
        # Load scaler and scaling info
        scaler, scaling_info = load_scaler_info()
        # Get scaling indices
        if isinstance(scaling_info, list):
            cols_to_scale = scaling_info
        elif isinstance(scaling_info, dict) and 'cols_to_scale' in scaling_info:
            cols_to_scale = scaling_info['cols_to_scale']
        else:
            raise ValueError("Invalid scaling_info format")
        
        # CRITICAL VALIDATION - Check if indices are within bounds
        if max(cols_to_scale) >= X.shape[1]:
            available_features = X.shape[1]
            needed_features = max(cols_to_scale) + 1
            raise ValueError(
                f"Feature dimension mismatch! "
                f"Data has {available_features} features, "
                f"but scaler expects feature index {max(cols_to_scale)}. "
                f"Need {needed_features} features total."
            )
        
        # Scale the data - direct numpy array operations
        X_scaled = X.copy()  # Create a copy to avoid modifying original
        features_to_scale = X_scaled[:, cols_to_scale]
        scaled_features = scaler.transform(features_to_scale)
        X_scaled[:, cols_to_scale] = scaled_features
        
        return X_scaled
        
    except Exception as e:
        print(f"❌ Error in scale_data: {e}")
        raise

def scale_all_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, cols_to_scale: list
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(X_train[:,cols_to_scale])
    
    # Save the scaling indices for future use
    scaling_info = {
        'cols_to_scale': cols_to_scale
    }

    # Creating copies to preserve all original data
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # Scaling data on cols to scale
    X_train_scaled[:, cols_to_scale] = scaler.transform(X_train[:,cols_to_scale])
    X_val_scaled[:, cols_to_scale] = scaler.transform(X_val[:,cols_to_scale])
    X_test_scaled[:, cols_to_scale] = scaler.transform(X_test[:,cols_to_scale])

    save_scaler_info(scaler, scaling_info)
    return X_train_scaled, X_val_scaled, X_test_scaled 

def log_target(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)
    y_test = np.log1p(y_test)
    return y_train, y_val, y_test

def log_target_eval(y: np.ndarray) -> np.ndarray:
    y_log = np.log1p(y)

    return y_log

def inverse_log1p(y_pred_log):
    """Convert log predictions back to original scale"""
    return np.expm1(y_pred_log)

def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int=72) -> tuple[np.ndarray, np.ndarray]:
    print(f'X: {len(X)}, y: {len(y)}')
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    
    if len(X) <= time_steps:
        raise ValueError(f"X length ({len(X)}) must be greater than time_steps ({time_steps})")

    if len(X) < time_steps * 2:
        print(f"⚠️ Warning: Limited data for sequence creation. X length: {len(X)}, time_steps: {time_steps}")

    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])

    return np.array(Xs), np.array(ys)

def create_LSTM_model(X_train_seq: np.ndarray, time_steps: int) -> Sequential:

    n_features = X_train_seq.shape[2] 
    model = Sequential([
        # First LSTM layer - capture basic temporal patterns
        LSTM(128, return_sequences=True, 
             dropout=0.2, recurrent_dropout=0.1,
             kernel_regularizer=l2(1e-4), 
             input_shape=(time_steps, n_features)),
        BatchNormalization(),

        # Second LSTM layer - capture higher-level dependencies
        LSTM(64, dropout=0.2, recurrent_dropout=0.1, 
             kernel_regularizer=l2(1e-4)),
        BatchNormalization(),

        # Dense layers for final prediction
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(1)
    ])   

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae'])
    
    model.summary()

    return model

def create_data_pipeline(X, y, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # For speed
    dataset = dataset.cache()  # Cache in memory if possible
    return dataset

def train_model(model: Sequential, X_train_seq: np.ndarray, y_train_seq: np.ndarray, X_val_seq: np.ndarray, y_val_seq: np.ndarray
                ) -> tuple[Sequential, dict]:
    early_stopping = EarlyStopping(
        patience=10,
        monitor='val_loss',
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        factor=0.5,
        patience=5,
        verbose=1
    )

    train_dataset = create_data_pipeline(X_train_seq, y_train_seq)
    val_dataset = create_data_pipeline(X_val_seq, y_val_seq)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return model, history

def visualize_pred(y_test, y_pred, n_points=1000):
    plt.figure(figsize=(15,10))

    plt.subplot(2, 1, 1)
    plt.plot(y_test[-n_points:], label='Actual', lw=2, alpha=0.8)
    plt.plot(y_pred[-n_points:], label='Predicted', alpha=0.7)
    plt.title('LSTM Forecast vs Actual (Test set - Last {} points)'.format(n_points))
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    errors = y_test[-n_points:] - y_pred[-n_points:]
    plt.plot(errors, label='Prediction Errors', color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Prediction Errors')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_prediction_sequences(X_scaled: np.ndarray, time_steps: int=72) -> np.ndarray:
    """
    Create sequences for prediction (without target variable)
    """
    sequences = []
    for i in range(len(X_scaled) - time_steps + 1):
        seq = X_scaled[i:i + time_steps]
        sequences.append(seq)
    
    return np.array(sequences)

def create_eval_sequences(y: np.ndarray, time_steps: int=72) -> np.ndarray:
    ys = []
    for i in range(len(y)-time_steps):
        ys.append(y[i + time_steps])

    return np.array(ys[time_steps:])

def predict_lstm(model: Sequential, X: pd.DataFrame, time_steps: int=72) -> np.ndarray:

    X = X.copy()

    # X = drop_cols(X)

    X_array = X.to_numpy()

    X_scaled = scale_data(X_array)

    # Create sequences from scaled features and log-transformed target
    X_seq = create_prediction_sequences(X_scaled, time_steps)

    # Predict on sequenced input and inverse transform the log predictions
    y_pred_log = model.predict(X_seq).flatten()
    y_pred = inverse_log1p(y_pred_log)

    return y_pred

def test_prediction(model: Sequential, X_test_seq: np.ndarray, y_test_seq: np.ndarray) -> dict:
    def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred) / y_true) * 100)

    y_pred_test = model.predict(X_test_seq).flatten()
    y_pred_test = inverse_log1p(y_pred_test)
    y_test_seq = inverse_log1p(y_test_seq)
    
    results = {
        "test_mae": mean_absolute_error(y_test_seq, y_pred_test),
        "test_rmse": root_mean_squared_error(y_test_seq, y_pred_test),
        "test_mape": mape(y_test_seq, y_pred_test)
    }

    for k,v in results.items():
        print(f'{k:<12}: {v:.3f}')

    visualize_pred(y_test_seq, y_pred_test)

    return results
        

def main():

    # Loading cleaned data
    df = pd.read_csv(os.path.join(DATA_FOLDER, 'worked/worked_dataset.csv'), parse_dates=['datetime'])
    df = df.sort_values('datetime')

    train_df, val_df, test_df = splitting_data(df) # Splitting data for model

    # Feature engineering with no data leakege
    train_df, scaler, pca = feature_engineering(train_df)
    val_df, _, _ = feature_engineering(val_df, scaler=scaler, pca=pca)
    test_df, _, _ = feature_engineering(test_df, scaler=scaler, pca=pca)
    train_df = drop_cols(train_df)
    val_df = drop_cols(val_df)
    test_df = drop_cols(test_df)

    # Preparing data for training the model
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_lstm(train_df, val_df, test_df, target_col='nat_demand')

    # Getting cols to scale
    cols_to_scale = get_cols_to_scale(train_df, 'nat_demand')

    # Scaling cols and log1p the target
    X_train_scaled, X_val_scaled, X_test_scaled = scale_all_data(X_train, X_val, X_test, cols_to_scale)

    y_train_log, y_val_log, y_test_log = log_target(y_train, y_val, y_test)

    # Creating sequences and checking shape
    time_steps = 168 # Lenght of sequences
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_log, time_steps)
    print('Train Seq: ',X_train_seq.shape, y_train_seq.shape)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_log, time_steps)
    print('Val Seq: ', X_val_seq.shape, y_val_seq.shape)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_log, time_steps)
    print('Test Seq: ', X_test_seq.shape, y_test_seq.shape)

    # Creating the model
    model = create_LSTM_model(X_train_seq, time_steps)

    # Training model
    model, history = train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq)

    # Predict
    results = test_prediction(model, X_test_seq, y_test_seq)

    # Save model, history, time steps and results
    time_steps ={
        'time_stpes': time_steps
    }
    save(model, history, results, time_steps)



if __name__ == '__main__':

    main()