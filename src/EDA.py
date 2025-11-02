import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
import os

def EDA(df: pd.DataFrame, data_folder, data_eda_folder, verbose):
    df = df.copy()

    df = df.sort_values('datetime')

    # Setting date as index
    df = df.set_index('datetime')

    # Checking structure
    print(df.shape)
    print(df.columns)
    print(df.head())

    # Checkinf any missing data
    print(df.isna().sum().sort_values(ascending=False))
    print(df.describe().T)

    # Total Demand start -> end
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df['nat_demand'], lw=1)
    plt.title('National Electricity Demand Over Time', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.grid(True)
    plt.savefig(os.path.join(data_eda_folder, "National_Electricity_Demand_Over_Time.png"))
    if verbose:
        plt.show()
    else:
        plt.close()


    # Electricity Demand Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Electricity Demand Analysis', fontsize=16)

    # Daily mean
    df['nat_demand'].resample('D').mean().plot(ax=axes[0, 0], title='Daily Average Demand', lw=1)

    # Weekly mean
    df['nat_demand'].resample('W').mean().plot(ax=axes[0, 1], title='Weekly Average Demand', lw=1)

    # Monthly mean
    df['nat_demand'].resample('ME').mean().plot(ax=axes[1, 0], title='Monthly Average Demand', lw=1)

    # Looking at the graphs we can see Covid makes the demand drop drastically
    # to make it work without any issues later i'll implement covid_period column
    df['covid_period'] = ((df.index >= '2020-03-01') & (df.index <= '2020-06-27')).astype(int) # Obtained dates searching online

    # COVID period visualization
    axes[1, 1].plot(df.index, df['nat_demand'], label='Demand', lw=1)
    axes[1, 1].fill_between(df.index, df['nat_demand'].min(), df['nat_demand'],
                        where=df['covid_period']==1, color='red', alpha=0.2, label='COVID Period')
    axes[1, 1].set_title('Electricity Demand with COVID Period Highlighted')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(data_eda_folder, "Electricity_Demand_Analysis.png"))

    if verbose:
        plt.show()
    else:
        plt.close()

    before = df.loc[(df.index < '2020-03-01'), 'nat_demand']
    during = df.loc[(df['covid_period'] == 1), 'nat_demand']

    print('Average demand before COVID: ', before.mean())
    print('Average demand during COVID: ', during.mean())
    print('Rrelative drop (%): ', (1- during.mean()/before.mean()) * 100)


    # Correlation between weather and demand
    weather_cols = [c for c in df.columns if 'T2M' in c or 'QV2M' in c or 'W2M' in c]
    corr = df[weather_cols + ['nat_demand']].corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix: Weather vs National Demand')

    plt.savefig(os.path.join(data_eda_folder, "Correlation_Matrix_Weather_vs_National_Demand.png"))

    if verbose:
        plt.show()
    else:
        plt.close()

    # Hourly and Weekly Patterns
    df['hour'] = df.index.hour              # type: ignore
    df['dayofweek'] = df.index.dayofweek    # type: ignore

    plt.figure(figsize=(12,4))
    sns.lineplot(data=df, x='hour', y='nat_demand', errorbar=None)
    plt.title('Average Demand by Hour of Day')

    plt.savefig(os.path.join(data_eda_folder, "Average_Demand_by_Hour_of_Day.png"))

    if verbose:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10,4))
    sns.barplot(data=df, x='dayofweek', y='nat_demand', errorbar=None)
    plt.title('Average Demand by Day of Week (0=Mon)')

    plt.savefig(os.path.join(data_eda_folder, "Average_Demand_by_Day_of_Week.png"))

    if verbose:
        plt.show()
    else:
        plt.close()

    # Trend & Seasonality Decomposition
    daily = df['nat_demand'].resample('D').mean()

    decomp = seasonal_decompose(daily, model='additive', period=365)
    decomp.plot()
    plt.suptitle('Seasonal Decomposition (Daily Demand)', fontsize=14)

    plt.savefig(os.path.join(data_eda_folder, "Seasonal_Decomposition.png"))

    if verbose:
        plt.show()
    else:
        plt.close()

    # Autocorrelation (ACF)
    autocorrelation_plot(df['nat_demand'])
    plt.title('Autocorrelation of Electricity Demand')
    
    plt.savefig(os.path.join(data_eda_folder, "Autocorrelation_of_Electricity_Demand.png"))

    if verbose:
        plt.show()
    else:
        plt.close()

    # Weather vs Demand Visualization
    plt.figure(figsize=(10,4))
    sns.scatterplot(x=df['T2M_toc'], y=df['nat_demand'], alpha=0.3)
    plt.title('Temperature (Toc) vs Electricity Demand')

    plt.savefig(os.path.join(data_eda_folder, "Temperature_(Toc)_vs_Electricity_Demand.png"))

    if verbose:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10,4))
    sns.scatterplot(x=df['T2M_san'], y=df['nat_demand'], alpha=0.3)
    plt.title('Temperature (San) vs Electricity Demand')

    plt.savefig(os.path.join(data_eda_folder, "Temperature_(San)_vs_Electricity_Demand.png"))
  
    if verbose:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10,4))
    sns.scatterplot(x=df['T2M_dav'], y=df['nat_demand'], alpha=0.3)
    plt.title('Temperature (Dav) vs Electricity Demand')

    plt.savefig(os.path.join(data_eda_folder, "Temperature_(Dav)_vs_Electricity_Demand.png"))
    
    if verbose:
        plt.show()
    else:
        plt.close()


    # Holidy Effects
    plt.figure(figsize=(10,4)) 
    sns.boxplot(data=df, x='school', y='nat_demand')
    plt.title('Demand Distribution: School Open vs Closed')

    plt.savefig(os.path.join(data_eda_folder, "Demand_Distribution_School_Open_vs_Closed.png"))

    if verbose:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10,4))
    sns.boxplot(data=df, x='Holiday_ID', y='nat_demand')
    plt.xticks(rotation=45)
    plt.title('Demand Distribution by Holiday_ID')

    plt.savefig(os.path.join(data_eda_folder, "Demand_Distribution_by_Holiday_ID.png"))
    
    if verbose:
        plt.show()
    else:
        plt.close()

    df = df.reset_index()
    df.to_csv(os.path.join(data_folder, 'worked/worked_dataset.csv'), index=False)

    return df

def main():
    # Root project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_eda_folder = os.path.join(project_root, 'reports/EDA')

    data_folder = os.path.join(project_root, 'data')

    # Loading cleaned data
    df = pd.read_csv(os.path.join(data_folder, 'worked/cleaned_continuous_data_set.csv'), parse_dates=['datetime'])

    df = EDA(df, data_folder, data_eda_folder, verbose=True)

if __name__ == '__main__':
        
    main()