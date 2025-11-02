import pandas as pd
import matplotlib.pyplot as plt
import os

def clean_data(df: pd.DataFrame, data_folder, data_prep_folder) -> pd.DataFrame:
    df = df.copy()

    # Examining the data
    print(df.shape)
    print(df.describe().T)
    print(df.info())
    print(df.head())

    # Sorting, basic cleaning, check duplicates
    df = df.sort_values('datetime').drop_duplicates(subset='datetime')
    print(f"Time span: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"Missing timestamps: {df['datetime'].isna().sum()}")

    # Checking uniform hourly frequency
    df = df.set_index('datetime')

    delta = df.index.to_series().diff().value_counts().head()
    print("Top time step frequencies:\n", delta)

    # Creating mean and range temperature columns
    df['temp_mean'] = df[['T2M_toc', 'T2M_san', 'T2M_dav']].mean(axis=1)
    df['temp_range'] = df[['T2M_toc', 'T2M_san', 'T2M_dav']].max(axis=1) - df[['T2M_toc', 'T2M_san', 'T2M_dav']].min(axis=1)

    # Checking Missing values
    print(df.isna().sum())

    # Basic sanity plots
    df['nat_demand'].head(672).plot(figsize=(12,8), title=f'Nat Demand Over Time {672/24} days', lw=1) # 4 weeks
    plt.savefig(os.path.join(data_prep_folder, "nat_demand.png"))
    plt.show()

    df.plot(x='temp_mean', y='nat_demand', kind='scatter', alpha=0.3)
    plt.savefig(os.path.join(data_prep_folder, "nat_demand_mean.png"))
    plt.show()

    df = df.reset_index()
    df.to_csv(os.path.join(data_folder, 'cleaned_continuous_data_set.csv'), index=False)
    
    return df

if __name__ == '__main__':
        
    # Root project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_prep_folder = os.path.join(project_root, 'reports/data_prep')

    data_folder = os.path.join(project_root, 'data/raw')

    save_folder = os.path.join(project_root, 'data/worked')

    # Getting all the data
    df = pd.read_csv(os.path.join(data_folder, 'continuous_dataset.csv'), parse_dates=['datetime'])
    
    df = clean_data(df, save_folder, data_prep_folder)