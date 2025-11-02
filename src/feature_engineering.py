import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def weather_pca(df: pd.DataFrame, scaler=None, pca=None, fit_pca=False) -> pd.DataFrame:
    # Seen correlation between these columns more then 90% in EDA
    # So instead of keeping 9 columns i'll create 3 using PCA
    # Standardize (for PCA)
    weather_cols = [c for c in df.columns if 'T2M' in c or 'QV2M' in c or 'W2M' in c]

    if fit_pca or scaler is None or pca is None:
        # Fit new scaler/PCA (for training data)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[weather_cols])
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(scaled)
    else:
        # Use pre-fitted scaler/PCA (for val/test data)
        scaled = scaler.transform(df[weather_cols])
        pca_features = pca.transform(scaled)

    df['weather_pc1'] = pca_features[:,0]
    df['weather_pc2'] = pca_features[:,1]
    df['weather_pc3'] = pca_features[:,2]
    

    # Dropping weather columns
    df = df.drop(columns=weather_cols)

    explained = pca.explained_variance_ratio_
    plt.figure(figsize=(6,4))
    plt.bar(range(1, len(explained)+1), explained, alpha=0.7, color='royalblue')
    plt.plot(range(1, len(explained)+1), np.cumsum(explained), 'r--', marker='o', label='Cumulative variance')
    plt.title('PCA Explained Variance Ratio')
    plt.xlabel('Principal Compnent')
    plt.ylabel('Variance Explained')
    plt.legend()

    return df

def feature_engineering(df: pd.DataFrame, data_folder: str|None=None, feature_engineering_folder: str|None=None, verbose=False, scaler=None, pca=None, fit_pca=False
                        ) -> tuple[pd.DataFrame, StandardScaler | None, PCA | None]:
    df = df.copy()
    
    df = weather_pca(df, scaler, pca, fit_pca)
    
    if feature_engineering_folder:
        plt.savefig(os.path.join(feature_engineering_folder, "PCA_Explained_Variance.png"))
    
    if verbose:
        plt.show()
    else:
        plt.close()

    # Temporal Features (lag, roll)
    df['lag_1'] = df['nat_demand'].shift(1)
    df['lag_24'] = df['nat_demand'].shift(24)
    df['lag_168'] = df['nat_demand'].shift(168)
    

    # Calender Features
    # Behavior
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['week'] = df['datetime'].dt.isocalendar().week.astype(int)
    df['day'] = df['datetime'].dt.day
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    # Cyclical encodings
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Ensuring no NaNs
    df[['weather_pc1', 'weather_pc2', 'weather_pc3']] = df[['weather_pc1', 'weather_pc2', 'weather_pc3']].fillna(0)

    # Diffrence vs seasonal
    df['diff_1'] = df['nat_demand'] - df['lag_1']
    df['diff_24'] = df['nat_demand'] - df['lag_24']
    df['diff_168'] = df['nat_demand'] - df['lag_168']

    # Weekend interactions
    df['hour_x_weekend'] = df['hour'] * df['is_weekend']
    df['hour_x_school'] = df['hour'] * df['school']

    df = df.dropna().reset_index(drop=True)

    if data_folder is not None:
        print(df.columns)
        print(df.info())
        print(df.describe().T)
        for col in df.columns:
            print(f'{col}: {df[col].min()} -> {df[col].max()}')
        df.to_csv(os.path.join(data_folder, 'worked/engineered_data.csv'), index=False)
    else:
        df = df.drop(columns=['month','week', 'day'], errors='ignore')


    return df, scaler, pca

if __name__ == '__main__':  
    # Root project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    feature_engineering_folder = os.path.join(project_root, 'reports/feature_engineering')

    data_folder = os.path.join(project_root, 'data')

    # Loading cleaned data
    df = pd.read_csv(os.path.join(data_folder, 'worked/worked_dataset.csv'), parse_dates=['datetime'])
    df = df.sort_values('datetime')

    df, _, _= feature_engineering(df, data_folder, feature_engineering_folder, verbose=True)