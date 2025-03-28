import os
import numpy as np
import pandas as pd
from tempCodeRunnerFile import process_dataset, train_and_evaluate  
from roi_detection import extract_roi_features
from sklearn.metrics import mean_squared_error, r2_score
from tempCodeRunnerFile import IRVideoProcessing, extract_excel_features  


dataset_folder = os.path.join(os.getcwd(), "dataset_new")
excel_path = os.path.join(os.getcwd(), "dataset", "DAQ_LW_Gypsum.xlsx")
roi = (100, 100, 200, 200)  # ROI for IR features extraction
i_spot_center = np.array([329, 331, 524])
j_spot_center = np.array([325, 400, 413])
m_array = np.array([2, 2, 2])
num_segments = 3

def load_global_features():
    """Extract only the global (and segmented) features plus excel features without ROI."""
    features_list = []

    for item in os.listdir(dataset_folder):
        item_path = os.path.join(dataset_folder, item)
        if os.path.isfile(item_path) and item.endswith(".mat"):
            folder_name = os.path.splitext(item)[0]
            mat_filepath = item_path
        elif os.path.isdir(item_path):
            folder_name = item
            mat_files = [f for f in os.listdir(item_path) if f.endswith(".mat")]
            if not mat_files:
                continue
            mat_filepath = os.path.join(item_path, mat_files[0])
        else:
            continue

        try:
            airflow_rate = float(item.split("fan_")[1].split("v")[0])
            # Use item_path as png_folder if directory, else dataset_folder
            png_folder = item_path if os.path.isdir(item_path) else dataset_folder
            ir_video = IRVideoProcessing(mat_filepath, png_folder)
            global_features = ir_video.extract_features(num_segments=num_segments, roi=roi)
            
            # code for Excel features if available
            if os.path.exists(excel_path):
                sheet_name = f"{airflow_rate}V"
                excel_features = extract_excel_features(excel_path, sheet_name)
            else:
                excel_features = {}
            
            combined = {**global_features, **excel_features, "airflow_rate": airflow_rate}
            features_list.append(combined)
        except Exception as e:
            print(f"Error processing {folder_name}: {e}")
            continue

    return pd.DataFrame(features_list)

def load_roi_features():
    """Extract only ROI-based features using the roi_detection module."""
    features_list = []
    for fname in os.listdir(dataset_folder):
        if fname.endswith(".mat"):
            mat_filepath = os.path.join(dataset_folder, fname)
            try:
                airflow_rate = float(fname.split("fan_")[1].split("v")[0])
                roi_feats = extract_roi_features(mat_filepath, i_spot_center, j_spot_center, m_array)
                roi_feats["airflow_rate"] = airflow_rate
                features_list.append(roi_feats)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                continue
    return pd.DataFrame(features_list)


combined_df = process_dataset(dataset_folder, excel_path, num_segments=num_segments, roi=roi,
                              i_spot_center=i_spot_center, j_spot_center=j_spot_center, m_array=m_array)
global_df = load_global_features()
roi_df = load_roi_features()

print("Combined Features DataFrame shape:", combined_df.shape)
print("Global Features DataFrame shape:", global_df.shape)
print("ROI Features DataFrame shape:", roi_df.shape)

# Evaluating each feature set using your training and evaluation pipeline.
print("\n--- Evaluating Combined Features ---")
train_and_evaluate(combined_df)

print("\n--- Evaluating Global Features Only ---")
train_and_evaluate(global_df)

print("\n--- Evaluating ROI Features Only ---")
train_and_evaluate(roi_df)
