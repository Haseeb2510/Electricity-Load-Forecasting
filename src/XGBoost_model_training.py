import numpy as np
import pandas as pd            
import xgboost as xgb
from xgboost import XGBRegressor           
from sklearn.metrics import mean_absolute_error, root_mean_squared_error  
from sklearn.model_selection import RandomizedSearchCV          
import matplotlib.pyplot as plt
import os
from prepare_data_for_models import splitting_data, prepare_data
from feature_engineering import feature_engineering
import joblib, json
import shap


# Root project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')
MODEL_FOLDER = os.path.join(PROJECT_ROOT, 'models/XGBoost')

def find_best_params(X_train: pd.DataFrame, y_train: pd.Series):
    params = {
        'max_depth': [5, 6, 7, 8],
        'min_child_weight': [1, 3, 5, 8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3, 0.5, 1],
        'learning_rate': [0.01, 0.03, 0.05]
    }
    search = RandomizedSearchCV(
        estimator=XGBRegressor(n_estimators=300, tree_method='hist'),
        param_distributions=params,
        scoring='neg_root_mean_squared_error',
        n_iter=20,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    print(search.best_params_)

def create_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=4000,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=8,
        subsample=0.8,
        early_stopping_rounds=300,
        colsample_bytree=0.8,
        reg_lambda=7.0,
        reg_alpha=0.5,
        tree_method='hist',  # Fast histogram-based algorithm
        gamma=0.5,
        objective='reg:squarederror',
        random_state=42
    )

    model.fit(
        X_train,y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    return model

def visualize_pred(y_test, y_pred, model, X_val, y_pred_test):
    plt.figure(figsize=(12,4))
    plt.plot(y_test.values[-1000:], label='Actual', lw=2)
    plt.plot(y_pred[-1000:], label='Predicted', lw=1.5)
    plt.title('XGBoost Forecast vs Actual - Test Set')
    plt.legend(); plt.show()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    shap.summary_plot(shap_values, X_val)

    residuals = y_test - y_pred_test
    plt.hist(residuals, bins=50)
    plt.title('Residual Distributaion')
    plt.show()

def predict_xgb(model: XGBRegressor, X: pd.DataFrame):
    y_pred = model.predict(X)
    return y_pred[-24:]

def predictions_val_test(model: XGBRegressor, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    def mape(y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / y_true) * 100) 
    # Predictions
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Metrics
    results = {
        "val_mae": mean_absolute_error(y_val, y_pred_val),
        "val_rmse": root_mean_squared_error(y_val, y_pred_val),
        "val_mape": mape(y_val, y_pred_val),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "test_rmse": root_mean_squared_error(y_test, y_pred_test),
        "test_mape": mape(y_test, y_pred_test)
    }

    for k,v in results.items():
        print(f'{k:<12}: {v:.3f}')

    visualize_pred(y_test, y_pred_test, model, X_val, y_pred_test)
    return results

def feature_importance(model: XGBRegressor):
    xgb.plot_importance(model, max_num_features=15, height=0.6)
    plt.title('Top 15 Features importance')
    plt.show()

def save_model(model: XGBRegressor, results: dict[str, float], model_folder):
    joblib.dump(model, os.path.join(model_folder, 'xgb_model.joblib'))
    with open(os.path.join(model_folder, 'xgb_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("✅ XGBoost model and metrics saved successfully.")

def load_model_xgb():
    try:
        model = joblib.load(os.path.join(MODEL_FOLDER, 'xgb_model.joblib'))
        print('✅ Model loaded successfully.')
    except Exception as e:
        print('Error: ', e)
        return e
    return model

def main():
    
    # Loading cleaned data
    df = pd.read_csv(os.path.join(DATA_FOLDER, 'worked/worked_dataset.csv'), parse_dates=['datetime'])
    df = df.sort_values('datetime')
        
    train_df, val_df, test_df = splitting_data(df) # Splitting data for model

    # Feature engineering with no data leakege
    train_df, scaler, pca = feature_engineering(train_df)
    val_df, _, _ = feature_engineering(val_df, scaler=scaler, pca=pca)
    test_df, _, _ = feature_engineering(test_df, scaler=scaler, pca=pca)


    # Preparing data for training the model
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(train_df, val_df, test_df, target_col='nat_demand')

    # FInd best params
    # find_best_params(X_train, y_train)

    # Creating and fitting the model
    model = create_model(X_train, y_train, X_val, y_val)

    # Predictions
    results = predictions_val_test(model, X_val, y_val, X_test, y_test)

    save_model(model, results, MODEL_FOLDER)

if __name__ == '__main__':
    main()