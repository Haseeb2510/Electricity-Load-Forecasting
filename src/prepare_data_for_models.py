import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def splitting_data(df: pd.DataFrame,
                cutoff_val: str | None=None,                                                                                # Date as string
                cutoff_test: str | None=None,                                                                               # Date as string
                split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                max_lag: int=168
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    if cutoff_val and cutoff_test:
        #  --- Manual date cutoffs ---
        train_df = df[df['datetime'] < cutoff_val]
        val_df = df[(df['datetime'] >= cutoff_val) & (df['datetime'] < cutoff_test)]
        test_df = df[df['datetime'] >= cutoff_test]
        

    else:
        # --- Automatic ratio split ---
        n = len(df)
        train_end = int(n * split_ratios[0]) 
        val_end = int(n * (split_ratios[0] + split_ratios[1])) 

        # Dates at those splits points (for reference)
        print("Train dates     :", df['datetime'].iloc[0], "→", df['datetime'].iloc[train_end - 1])
        print("Validation dates:", df['datetime'].iloc[train_end], "→", df['datetime'].iloc[val_end - 1])
        print("Test dates      :", df['datetime'].iloc[val_end], "→", df['datetime'].iloc[-1])

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

    print(f"Train: {train_df['datetime'].min()} → {train_df['datetime'].max()}  ({len(train_df)} rows)")
    print(f"Valid: {val_df['datetime'].min()}  → {val_df['datetime'].max()}  ({len(val_df)} rows)")
    print(f"Test : {test_df['datetime'].min()}  → {test_df['datetime'].max()}  ({len(test_df)} rows)")

    plt.figure(figsize=(10,4))
    plt.plot(train_df['datetime'], train_df['nat_demand'], label='Train')
    plt.plot(val_df['datetime'], val_df['nat_demand'], label='Val')
    plt.plot(test_df['datetime'], test_df['nat_demand'], label='Test')
    plt.title('Train vs Val vs Test Split')
    plt.show()

    return train_df, val_df, test_df

def splitting_data_eval(df: pd.DataFrame, horizon: int=336) -> tuple[pd.DataFrame, pd.Series]:
    
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    # Dropping redundant data
    df = df.drop(columns=['month','week', 'day'], errors='ignore')

    # Get the last horizon hours for prediction (features only)
    prediction_data  = df[-horizon:].copy()

    print(f"Test : {prediction_data ['datetime'].min()}  → {prediction_data ['datetime'].max()}  ({len(prediction_data )} rows)")

    actual_values = prediction_data['nat_demand']  # if we want to compare later

    # Visualization of what you're predicting
    plt.figure(figsize=(10, 4))
    plt.plot(prediction_data ['datetime'], prediction_data ['nat_demand'], label='Actual (to compare with predictions)')
    plt.title('Data for Prediction (Last 100 Hours)')
    plt.legend()
    plt.show()

    return prediction_data, actual_values

def prepare_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str
                 ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    target_col = target_col

    # Features (drop datetime & target)
    feature_cols = [col for col in train_df.columns if col not in ['datetime', target_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test 

def prepare_data_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Features (drop datetime & target)
    feature_cols = [col for col in train_df.columns if col not in ['datetime', target_col]]

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_col].to_numpy()

    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df[target_col].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df[target_col].to_numpy()

    return X_train, y_train, X_val, y_val, X_test, y_test    

def get_cols_to_scale(df: pd.DataFrame,
                      target_col: str | None = None,
                      min_range: float=5.0) -> list:
    # Practical approach to identify columns for scaling.
    df = df.drop(columns=['datetime', target_col], errors='ignore')

    cols_to_scale = []
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    # Never scale
    never_scale = []
    if target_col:
        never_scale.append(target_col)
    
    for i, col in enumerate(numeric_cols) :
        if col in never_scale:
            continue
        
        col_range = df[col].max() - df[col].min()

        # Scale if range is significant
        if col_range > min_range:
            cols_to_scale.append(i)

    print('Cols to scale: ', cols_to_scale)

    return cols_to_scale

if __name__ == '__main__':
    # Root project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_folder = os.path.join(project_root, 'data')

    # Loading cleaned data
    df = pd.read_csv(os.path.join(data_folder, 'worked/engineered_data.csv'), parse_dates=['datetime'])
    df = df.sort_values('datetime')

    split_ratios = (0.75,0.15,0.15)
    train_df, val_df, test_df = splitting_data(df, split_ratios=split_ratios)