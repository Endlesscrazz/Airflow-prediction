import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from airflow_model import process_dataset  

def perform_eda(df):
    if df.empty:
        print("The DataFrame is empty. Please check your dataset and extraction process.")
        return

    # Display first few rows, info, and summary statistics
    print("DataFrame Head:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()
    print("\nStatistical Summary:")
    print(df.describe())

    # Plot histograms for all numerical features
    df.hist(figsize=(20, 20))
    plt.suptitle("Histograms of all features", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Plot a correlation matrix
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Plot a scatter matrix for a visual overview of feature relationships
    scatter_matrix(df, figsize=(15, 15), diagonal='kde')
    plt.suptitle("Scatter Matrix of Features", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define your dataset folder and Excel file path (adjust if necessary)
    dataset_folder = os.path.join(os.getcwd(), "dataset_new")
    excel_path = os.path.join(os.getcwd(), "dataset_new", "DAQ_LW_Gypsum.xlsx")
    
    # Define an example ROI and the number of segments (update these values as needed)
    roi = (100, 100, 200, 200)  # (x, y, width, height)
    num_segments = 1

    # Reuse your existing function to process the dataset and extract features.
    df = process_dataset(dataset_folder, excel_path, num_segments=num_segments, roi=roi)

    if df.empty:
        print("No data processed. Check your dataset folder and Excel file.")
    else:
        print("Extracted Feature DataFrame:")
        print(df.head())
        # Perform the exploratory data analysis on the extracted features.
        perform_eda(df)
