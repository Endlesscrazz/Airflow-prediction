#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import scipy.io

# Import functions and classes from your modules.
import airflow_model
import roi_detection

# Reload modules (useful during development)
importlib.reload(airflow_model)
importlib.reload(roi_detection)

# Import specific functions/classes from the modules.
from airflow_model import IRVideoProcessing, parse_delta_T, parse_airflow_rate, ROI_CONFIGS
from roi_detection import extract_roi_features

# =============================================================================
# User-Defined Settings (Flags)
# =============================================================================
USE_ROI_FEATURES = True         # Set to False to extract only global features.
DATASET_FOLDER = os.path.join(os.getcwd(), "dataset_new")
EXCEL_PATH = os.path.join(os.getcwd(), "dataset_new", "DAQ_LW_Gypsum.xlsx")  # Optional; can be None.
NUM_SEGMENTS = 1                # Global feature extraction segments.
GLOBAL_ROI = None               # Use entire frame for global features (i.e. no cropping).

# =============================================================================
# Dataset Processing Function
# =============================================================================
def process_dataset_custom(dataset_folder, excel_path, num_segments, global_roi, use_roi):
    """
    Processes the dataset folder by iterating through subfolders. For each .mat file
    in a subfolder (expected to be named like "FanPower_1.8V", etc.), global features
    are extracted from the entire frame (if global_roi is None) using IRVideoProcessing.
    If use_roi is True, ROI features are also extracted using the ROI_CONFIGS dictionary.
    
    Args:
        dataset_folder (str): Root folder containing subfolders with .mat files.
        excel_path (str): Path to an Excel file with additional features (optional).
        num_segments (int): Number of segments for global feature extraction.
        global_roi (tuple or None): ROI for global extraction. If None, the entire frame is used.
        use_roi (bool): Flag indicating whether to include ROI features.
    
    Returns:
        pd.DataFrame: DataFrame with combined features.
    """
    features_list = []
    
    # Iterate through each subfolder.
    for folder_name in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        try:
            airflow_rate = parse_airflow_rate(folder_name)
        except Exception as e:
            print(f"Skipping folder '{folder_name}': {e}")
            continue
        
        # List .mat files in the current subfolder.
        mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
        if not mat_files:
            print(f"No .mat files found in folder: {folder_name}")
            continue
        
        for mat_file in mat_files:
            mat_filepath = os.path.join(folder_path, mat_file)
            try:
                # Global features extraction using the entire frame.
                ir_video = IRVideoProcessing(mat_filepath, png_folder=folder_path)
                global_feats = ir_video.extract_features(num_segments=num_segments, roi=global_roi)
                if global_feats is None:
                    print(f"Global features extraction failed for {mat_file}")
                    continue
                
                # Parse delta T from the .mat file name.
                delta_T = parse_delta_T(mat_file)
                
                combined_feats = {}
                combined_feats.update(global_feats)
                
                # If ROI features are requested, add them.
                if use_roi:
                    if airflow_rate in ROI_CONFIGS:
                        cfg = ROI_CONFIGS[airflow_rate]
                        i_spot_center = np.array(cfg["i_spot_center"])
                        j_spot_center = np.array(cfg["j_spot_center"])
                        m_array_config = np.array(cfg["m_array"])
                        roi_feats = extract_roi_features(mat_filepath, i_spot_center, j_spot_center, m_array_config)
                        if roi_feats is not None:
                            combined_feats.update(roi_feats)
                        else:
                            print(f"ROI features extraction failed for {mat_file}")
                    else:
                        print(f"No ROI config found for airflow_rate {airflow_rate} in folder {folder_name}")
                
                # Add metadata.
                combined_feats["airflow_rate"] = airflow_rate
                combined_feats["delta_T"] = delta_T
                
                # (Optional: You can merge Excel features here if desired.)
                
                features_list.append(combined_feats)
            except Exception as e:
                print(f"Error processing {mat_file} in folder {folder_name}: {e}")
                continue
    return pd.DataFrame(features_list)

# =============================================================================
# Main Data Exploration Routine
# =============================================================================
def main():
    print(f"Dataset folder: {DATASET_FOLDER}")
    print(f"Using ROI features: {USE_ROI_FEATURES}")
    print(f"Global features extraction: using entire frame (no ROI cropping)")
    print(f"Number of segments: {NUM_SEGMENTS}")
    
    # Process the dataset.
    df_features = process_dataset_custom(DATASET_FOLDER, EXCEL_PATH, NUM_SEGMENTS, GLOBAL_ROI, USE_ROI_FEATURES)
    
    if df_features.empty:
        print("No features extracted. Please check your dataset folder and settings.")
        return
    
    # Display basic information.
    print("\nExtracted Features DataFrame:")
    print(df_features.head())
    print("Shape:", df_features.shape)
    print("Columns:", df_features.columns.tolist())
    
    # Plot a correlation heatmap (excluding the target 'airflow_rate').
    if "airflow_rate" in df_features.columns:
        corr = df_features.drop(columns=["airflow_rate"]).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of Extracted Features")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
