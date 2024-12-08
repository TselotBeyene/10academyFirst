import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from windrose import WindroseAxes

# Function to clean the dataset
def clean_dataset(dataset, name, output_dir):
    print(f"\nCleaning dataset for {name}...")

    # Check for missing values
    missing_values = dataset.isnull().sum()
    print(f"Missing values in {name} dataset:\n{missing_values}\n")

    # Checking for unexpected columns
    print(f"Columns in {name} dataset: {dataset.columns}")

    # Drop the 'Comments' column if it's entirely null
    if 'Comments' in dataset.columns:
        if dataset['Comments'].isnull().all():
            print(f"Dropping 'Comments' column in {name} because it's entirely null.")
            dataset.drop(columns=['Comments'], inplace=True)

    # Fill missing values with column mean
    dataset.fillna({
        'GHI': dataset['GHI'].mean(),
        'DNI': dataset['DNI'].mean(),
        'Tamb': dataset['Tamb'].mean(),
        'WS': dataset['WS'].mean(),
        'RH': dataset['RH'].mean()
    }, inplace=True)

    # Handling negative values in relevant columns
    dataset.loc[dataset['Tamb'] < 0, 'Tamb'] = dataset['Tamb'].mean()
    dataset.loc[dataset['GHI'] < 0, 'GHI'] = 0
    dataset.loc[dataset['DNI'] < 0, 'DNI'] = 0

    # Clamping GHI and DNI values to 2000
    dataset.loc[dataset['GHI'] > 2000, 'GHI'] = 2000
    dataset.loc[dataset['DNI'] > 2000, 'DNI'] = 2000

    # Save cleaned dataset to output directory
    cleaned_file = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_cleaned.csv")
    dataset.to_csv(cleaned_file, index=False)
    print(f"Cleaned dataset for {name} saved at {cleaned_file}\n")

    return dataset

# Function to perform analysis on a dataset
def analyze_dataset(data, name, output_dir):
    print(f"\nAnalyzing dataset for {name}...")

    # Convert Timestamp to datetime and set as index
    if data['Timestamp'].isnull().any():
        print(f"Warning: Missing 'Timestamp' values in {name}, dropping these rows.")
        data = data.dropna(subset=['Timestamp'])
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
    data.set_index('Timestamp', inplace=True)

    # Recheck for NaT in index
    print(f"Rechecking NaT values in index after cleaning: {data.index.isnull().sum()}")

    # Solar Radiation Over Time
    data[['GHI', 'DNI', 'DHI']].plot(figsize=(10, 6))
    plt.title(f"Solar Radiation (GHI, DNI, DHI) Over Time - {name}")
    plt.ylabel("Irradiance (W/m²)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_solar_radiation_over_time.png"))
    plt.close()

    # Monthly Averages of Solar Radiation
    data['Month'] = data.index.month
    monthly_avg = data.groupby('Month')[['GHI', 'DNI', 'DHI']].mean()
    monthly_avg.plot(kind='bar', figsize=(10, 6))
    plt.title(f"Monthly Averages of Solar Radiation - {name}")
    plt.ylabel("Irradiance (W/m²)")
    plt.xlabel("Month")
    plt.xticks(ticks=range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_monthly_averages_solar_radiation.png"))
    plt.close()

    # Correlation Matrix
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f"Correlation Matrix - {name}")
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_correlation_matrix.png"))
    plt.close()

    # Histograms of Key Variables
    columns_to_plot = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS']
    data[columns_to_plot].hist(bins=20, figsize=(12, 8), edgecolor='black')
    plt.suptitle(f"Histograms of Key Variables - {name}", y=0.95)
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_histograms_key_variables.png"))
    plt.close()

    # Z-Score Analysis: Outlier Detection
    data_zscore = data.select_dtypes(include=np.number)
    z_scores = np.abs(zscore(data_zscore))
    outliers = (z_scores > 3).sum(axis=0)
    print(f"Number of Outliers per Column (Z-Score > 3) for {name}:\n{outliers}")

    # Save Z-Score outliers to CSV
    outliers_file = os.path.join(output_dir, f"{name.lower()}_z_score_outliers.csv")
    outliers.to_csv(outliers_file)
    print(f"Z-score outliers for {name} saved at {outliers_file}\n")

# Main script
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
datasets = {
    "Benin": pd.read_csv("data/benin-malanville.csv"),
    "Sierra Leone": pd.read_csv("data/sierraleone-bumbuna.csv"),
    "Togo": pd.read_csv("data/togo-dapaong_qc.csv")
}

# Clean and analyze each dataset
for name, dataset in datasets.items():
    cleaned_data = clean_dataset(dataset, name, output_dir)
    analyze_dataset(cleaned_data, name, output_dir)

print("\nAll datasets processed successfully!")
