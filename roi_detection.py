#!/usr/bin/env python
import os
import re
import numpy as np
import scipy.io
import pandas as pd

def parse_airflow_rate(foldername):
    """
    Extract the airflow rate label from the folder name.
    Args:
        foldername (str): Example: "FanPower_1.6V".
    Returns:
        float: Extracted airflow rate.
    """
    match = re.search(r'FanPower_(\d+(\.\d+)?)V', foldername, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not parse airflow rate in folder name: {foldername}")

# --- NEW FUNCTION: Extract delta T from MATLAB file name ---
def parse_delta_T(filename):
    """
    Extract delta T (temperature difference) from the MATLAB file name.
    Args:
        filename (str): Example: "temp_2025-3-7-19-45-8_21.4_35_13.6_mat"
    Returns:
        float: Extracted delta T value.
    """
    tokens = filename.split('_')
    try:
        # Assumes the second-to-last token is the delta T value.
        delta_T = float(tokens[-2])
        return delta_T
    except Exception as e:
        raise ValueError(f"Could not parse delta T from filename {filename}: {e}")

def extract_roi_features(mat_filepath,i_spot_center, j_spot_center, m_array):
    """
    Extract ROI features from a given .mat file by sampling fixed spots.
    For each spot, this function extracts a window (of size determined by m_array)
    from each frame and computes the mean temperature over that window.
    It then summarizes the time-series by computing its overall mean and std.
    
    Args:
        mat_filepath (str): Path to the .mat file.
        i_spot_center (np.array): Array of x-coordinates for spot centers.
        j_spot_center (np.array): Array of y-coordinates for spot centers.
        m_array (np.array): Half-window sizes for each spot.
        
    Returns:
        dict: A dictionary of ROI-based features.
    """
    # Loading with squeeze_me=True to remove unnecessary dimensions
    try:
        
        data_struct = scipy.io.loadmat(mat_filepath, squeeze_me=True)
        TempFrames = data_struct['TempFrames']

        # Debugging: Print the type and shape of TempFrames
        #print(f"[DEBUG] Raw TempFrames type: {type(TempFrames)}, shape: {TempFrames.shape}")

        # Handle different cases for TempFrames
        if TempFrames.ndim == 1 or (TempFrames.dtype == object):
            # If TempFrames is a 1D array (of 2D frames), stack them into a 3D array
            TempFrames = np.stack([np.asarray(frame) for frame in TempFrames])
        elif TempFrames.ndim == 2:
            # If TempFrames is a 2D array, reshape it into a 3D array with 1 frame
            TempFrames = TempFrames[np.newaxis, :, :]
        elif TempFrames.ndim == 3:
            # If TempFrames is already a 3D array, do nothing
            pass
        else:
            raise ValueError(f"Unexpected shape for TempFrames: {TempFrames.shape}")


        # Debugging: Print the final shape of TempFrames
        #print(f"[DEBUG] Final TempFrames shape: {TempFrames.shape}")

        nframe, nx, ny = TempFrames.shape
        num_spots = len(i_spot_center)
        T_spot = np.zeros((num_spots, nframe, 2))

        for iframe in range(nframe):
            raw_T = TempFrames[iframe, :, :]
            for i_sample in range(num_spots):
                m = m_array[i_sample]
                # May need to clip ROI coordinates to image bounds
                i_start = max(i_spot_center[i_sample] - m, 0)
                i_end = min(i_spot_center[i_sample] + m + 1, ny)
                j_start = max(j_spot_center[i_sample] - m, 0)
                j_end = min(j_spot_center[i_sample] + m + 1, nx)

                sub_region = raw_T[j_start:j_end, i_start:i_end]

                if sub_region.size == 0:
                    mean_val = np.nan
                else:
                    mean_val = np.mean(sub_region)
                    
                time_val = iframe / 3.0  # Adjust as needed
                T_spot[i_sample, iframe, 0] = time_val
                T_spot[i_sample, iframe, 1] = mean_val

        features = {}
        for i_sample in range(num_spots):
            sample_values = T_spot[i_sample, :, 1]
            features[f"roi_sample_{i_sample+1}_mean"] = np.mean(sample_values)
            features[f"roi_sample_{i_sample+1}_std"] = np.std(sample_values)
        return features
    except Exception as e:
        print(f"Error in extract_roi_features: {e}")
        return None

def process_dataset(dataset_folder, i_spot_center, j_spot_center, m_array):
    """
    Process all .mat files in a given folder (dataset_new) to extract ROI features.
    
    Args:
        dataset_folder (str): Path to the folder containing the .mat files.
        i_spot_center (np.array): Array of x-coordinates for ROI spots.
        j_spot_center (np.array): Array of y-coordinates for ROI spots.
        m_array (np.array): Half-window sizes for the ROI spots.
    
    Returns:
        pd.DataFrame: A DataFrame where each row contains the extracted ROI features and the airflow rate.
    """
    roi_configs = {
    "FanPower_1.8V": {"i_spot_center": [329, 331, 524],
                       "j_spot_center": [325, 400, 413],
                       "m_array": [2, 2, 2]},
    "FanPower_2.4V": {"i_spot_center": [300, 320, 510],
                       "j_spot_center": [310, 380, 400],
                       "m_array": [3, 3, 3]}
    # Add other configurations as needed
}

    features_list = []
    # Loop directly over .mat files in the dataset folder
    for fname in os.listdir(dataset_folder):
        if fname.endswith(".mat"):
            mat_filepath = os.path.join(dataset_folder, fname)
            print(f"Processing file: {fname}")
            try:
                airflow_rate = parse_airflow_rate(fname)
                roi_features = extract_roi_features(mat_filepath, i_spot_center, j_spot_center, m_array)
                roi_features["airflow_rate"] = airflow_rate
                features_list.append(roi_features)
            except Exception as e:
                print(f"Error processing file {fname}: {e}")
                continue
    return pd.DataFrame(features_list)

if __name__ == "__main__":
    # Set the folder containing the 10 .mat files
    dataset_folder = os.path.join(os.getcwd(), "dataset_new")
    
    # Define sample ROI parameters (adjust based on your data)
    i_spot_center = [311, 300, 400]  # Column centers
    j_spot_center = [252, 300, 400]  # Row centers
    m_array = [2, 4, 4]     
    
    # Process the dataset to extract ROI features
    df_roi = process_dataset(dataset_folder, i_spot_center, j_spot_center, m_array)
    print("Extracted ROI Feature DataFrame:")
    print(df_roi)
    
    # Now, df_roi can be saved or passed directly as part of your ML input vector in your main pipeline.
