import os
import re
import numpy as np
import pandas as pd
import roi_detection
import importlib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from scipy.io import loadmat
from skimage import io
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import SimpleImputer  # <-- New import for imputation

# Reload your roi_detection module if needed
importlib.reload(roi_detection)
from roi_detection import extract_roi_features

# --------------------------
# MODEL SELECTION FLAGS
# --------------------------
use_rf = True
use_lr = False
use_gb = True

# --------------------------
# ROI CONFIGURATIONS
# --------------------------
ROI_CONFIGS = {
    1.6: {
        "i_spot_center": [340, 300, 400],
        "j_spot_center": [305, 300, 400],
        "m_array": [2, 4, 4]
    },
    1.8: {
        "i_spot_center": [340, 300, 400],
        "j_spot_center": [305, 300, 400],
        "m_array": [2, 4, 4]
    },
    2.1: {
        "i_spot_center": [340, 300, 400],
        "j_spot_center": [305, 300, 400],
        "m_array": [2, 4, 4]
    },
    2.4: {
        "i_spot_center": [311, 300, 400],
        "j_spot_center": [252, 300, 400],
        "m_array": [2, 4, 4]
    }
}

# --------------------------
# 1) Parse Airflow Rate from Folder Name
# --------------------------
def parse_airflow_rate(folder_name):
    """
    Extract the numeric airflow rate (fan voltage) from a folder name like 'FanPower_1.6V'.
    Returns float if found, otherwise raises ValueError.
    """
    match = re.search(r'FanPower_(\d+(\.\d+)?)V', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not parse airflow rate from folder name: {folder_name}")

# --------------------------
# 2) Parse Delta T from .mat File Name
# --------------------------
def parse_delta_T(mat_filename):
    """
    Extract delta T from the .mat file name.
    Example: 'temp_2025-3-7-19-14-21_21.4_35_13.6_.mat' -> 13.6
    We'll look for the second-to-last token before '.mat'.
    """
    base_name = mat_filename.replace('.mat', '')
    tokens = base_name.split('_')
    if len(tokens) < 2:
        return None
    dt_str = tokens[-2] if tokens[-1] == '' else tokens[-1]
    dt_str = dt_str.strip('_')
    try:
        return float(dt_str)
    except ValueError:
        print(f"Warning: Could not parse delta T from {mat_filename}, got '{dt_str}'")
        return None

# --------------------------
# IRVideoProcessing Class
# --------------------------
class IRVideoProcessing:
    def __init__(self, mat_filepath, png_folder=None):
        self.mat_filepath = mat_filepath
        self.png_folder = png_folder
        self.data = self.load_mat_data()
        self.png_frames = self.load_png_frames() if png_folder else None
        self.spatial_gradient_data = None
        self.temporal_gradient_data = None

    def load_mat_data(self):
        try:
            mat_data = loadmat(self.mat_filepath, squeeze_me=True)
            frames = mat_data["TempFrames"]
            if frames.ndim == 1 or (frames.dtype == object):
                frames = np.stack([np.asarray(frame) for frame in frames])
            elif frames.ndim == 2:
                frames = frames[np.newaxis, :, :]
            elif frames.ndim == 3:
                pass
            else:
                raise ValueError(f"Unexpected shape for TempFrames: {frames.shape}")
            frames = frames / 255.0
            return frames
        except Exception as e:
            print(f"Error loading IR video data: {e}")
            return None

    def load_png_frames(self):
        try:
            png_files = sorted([f for f in os.listdir(self.png_folder) if f.endswith(".png")])
            if not png_files:
                return None
            frames = [io.imread(os.path.join(self.png_folder, f), as_gray=True) for f in png_files]
            frames = np.array(frames)
            frames = frames / 255.0
            return frames
        except Exception as e:
            print(f"Error loading .png frames: {e}")
            return None

    def compute_spatial_gradient(self, roi=None):
        if self.data is None:
            return None
        gradients = []
        for frame in self.data:
            if roi is not None:
                frame = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            gx = np.gradient(frame, axis=0)
            gy = np.gradient(frame, axis=1)
            grad_mag = np.sqrt(gx**2 + gy**2)
            gradients.append(grad_mag)
        self.spatial_gradient_data = np.array(gradients)
        return self.spatial_gradient_data

    def compute_temporal_gradient(self, frame_step=1, roi=None):
        if self.data is None:
            return None
        num_frames = self.data.shape[0]
        if roi is not None:
            temp_grad = np.zeros((num_frames, roi[3], roi[2]))
        else:
            temp_grad = np.zeros_like(self.data)
        for i in range(num_frames - frame_step):
            f1 = self.data[i]
            f2 = self.data[i + frame_step]
            if roi is not None:
                f1 = f1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                f2 = f2[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            temp_grad[i] = np.abs(f2 - f1)
        self.temporal_gradient_data = temp_grad
        return self.temporal_gradient_data

    def extract_features(self, num_segments=1, roi=None):
        try:
            data_used = self.data[:, roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] if roi else self.data
            features = {}
            if num_segments == 1:
                frame_max = np.max(data_used, axis=(1, 2))
                frame_mean = np.mean(data_used, axis=(1, 2))
                frame_std = np.std(data_used, axis=(1, 2))
                features.update({
                    "overall_max": np.max(frame_max),
                    "overall_mean": np.mean(frame_mean),
                    "overall_std": np.mean(frame_std)
                })
                spatial_grad = self.compute_spatial_gradient(roi=roi)
                avg_spatial = np.mean(spatial_grad, axis=(1, 2))
                features.update({
                    "mean_spatial_grad": np.mean(avg_spatial),
                    "std_spatial_grad": np.std(avg_spatial)
                })
                temporal_grad = self.compute_temporal_gradient(roi=roi)
                valid_temp = temporal_grad[:-1]
                avg_temp = np.mean(valid_temp, axis=(1, 2))
                features.update({
                    "mean_temp_grad": np.mean(avg_temp),
                    "std_temp_grad": np.std(avg_temp)
                })
                if self.png_frames is not None:
                    png_used = self.png_frames[:, roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] if roi else self.png_frames
                    features["mean_png_intensity"] = np.mean(png_used)
                return features
            else:
                num_frames = data_used.shape[0]
                segment_length = num_frames // num_segments
                for seg in range(num_segments):
                    start = seg * segment_length
                    end = num_frames if seg == num_segments - 1 else (seg + 1) * segment_length
                    seg_data = data_used[start:end]
                    seg_max = np.max(seg_data, axis=(1, 2))
                    seg_mean = np.mean(seg_data, axis=(1, 2))
                    seg_std = np.std(seg_data, axis=(1, 2))
                    features[f"seg{seg+1}_overall_max"] = np.max(seg_max)
                    features[f"seg{seg+1}_overall_mean"] = np.mean(seg_mean)
                    features[f"seg{seg+1}_overall_std"] = np.mean(seg_std)

                    seg_spatial = []
                    for frame in seg_data:
                        gx = np.gradient(frame, axis=0)
                        gy = np.gradient(frame, axis=1)
                        seg_spatial.append(np.sqrt(gx**2 + gy**2))
                    seg_spatial = np.array(seg_spatial)
                    seg_avg_spatial = np.mean(seg_spatial, axis=(1, 2))
                    features[f"seg{seg+1}_mean_spatial_grad"] = np.mean(seg_avg_spatial)
                    features[f"seg{seg+1}_std_spatial_grad"] = np.std(seg_avg_spatial)

                    seg_temp = np.zeros_like(seg_data)
                    if seg_data.shape[0] > 1:
                        seg_temp[:-1] = np.abs(seg_data[1:] - seg_data[:-1])
                        seg_valid_temp = seg_temp[:-1]
                        seg_avg_temp = np.mean(seg_valid_temp, axis=(1, 2))
                        features[f"seg{seg+1}_mean_temp_grad"] = np.mean(seg_avg_temp)
                        features[f"seg{seg+1}_std_temp_grad"] = np.std(seg_avg_temp)
                    else:
                        features[f"seg{seg+1}_mean_temp_grad"] = 0
                        features[f"seg{seg+1}_std_temp_grad"] = 0

                    if self.png_frames is not None:
                        png_used = self.png_frames[:, roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] if roi else self.png_frames
                        png_seg = png_used[start:end]
                        features[f"seg{seg+1}_mean_png_intensity"] = np.mean(png_seg)
            return features
        except Exception as e:
            print(f"Error in extract_features: {e}")
            return None

# --------------------------
# Optional: Excel features if needed
# --------------------------
def extract_excel_features(excel_path, sheet_name):
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error reading Excel file {excel_path} (sheet: {sheet_name}): {e}")
        return None
    return {
        "mean_rtd": df["RTD (°C)"].mean(),
        "mean_dp": df["ΔP (Pa)"].mean(),
        "var_rtd": df["RTD (°C)"].var(),
        "var_dp": df["ΔP (Pa)"].var(),
        "rate_of_change_rtd": np.mean(np.diff(df["RTD (°C)"])),
        "rate_of_change_dp": np.mean(np.diff(df["ΔP (Pa)"]))
    }

# --------------------------
# 3) Main Process Dataset
# --------------------------
def process_dataset(dataset_folder, excel_path, num_segments=1, roi=None):
    """
    dataset_folder structure:
      dataset_new/
        FanPower_1.6V/
          temp_2025-3-7-19-14-21_21.4_35_13.6_.mat
          ...
        FanPower_1.8V/
          ...
    We parse the airflow rate from 'FanPower_1.6V' -> 1.6,
    parse the delta T from the .mat file name -> e.g. 13.6,
    then select the corresponding ROI configuration based on ROI_CONFIGS.
    """
    features_list = []
    # Iterate through subfolders (e.g., FanPower_1.6V, FanPower_1.8V, etc.)
    for folder_name in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        try:
            airflow_rate = parse_airflow_rate(folder_name)
        except ValueError as e:
            print(f"Skipping folder '{folder_name}' - {e}")
            continue

        if airflow_rate not in ROI_CONFIGS:
            print(f"No ROI config found for airflow_rate={airflow_rate}, skipping folder.")
            continue
        roi_cfg = ROI_CONFIGS[airflow_rate]
        i_spot_center = np.array(roi_cfg["i_spot_center"])
        j_spot_center = np.array(roi_cfg["j_spot_center"])
        m_array = np.array(roi_cfg["m_array"])

        mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
        if not mat_files:
            print(f"No .mat files found in folder: {folder_name}")
            continue

        for mat_file in mat_files:
            mat_filepath = os.path.join(folder_path, mat_file)
            try:
                delta_T = parse_delta_T(mat_file)
                ir_video = IRVideoProcessing(mat_filepath, png_folder=folder_path)
                ir_features = ir_video.extract_features(num_segments=num_segments, roi=roi)
                sheet_name = f"{airflow_rate}V"
                if os.path.exists(excel_path):
                    excel_features = extract_excel_features(excel_path, sheet_name)
                    if excel_features is None:
                        excel_features = {}
                else:
                    excel_features = {}
                roi_features = extract_roi_features(mat_filepath, i_spot_center, j_spot_center, m_array)
                if roi_features is None:
                    continue
                combined_features = {
                    **ir_features,
                    **excel_features,
                    **roi_features,
                    "airflow_rate": airflow_rate,
                    "delta_T": delta_T
                }
                features_list.append(combined_features)
            except Exception as e:
                print(f"Error processing {mat_file} in folder {folder_name}: {e}")
                continue

    return pd.DataFrame(features_list)

# --------------------------
# 4) Train and Evaluate
# --------------------------
def train_and_evaluate(df):
    model_list = []
    if use_rf:
        model_list.append(("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42)))
    if use_lr:
        model_list.append(("LinearRegression", LinearRegression()))
    if use_gb:
        model_list.append(("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)))

    X = df.drop(columns=["airflow_rate"])
    y = df["airflow_rate"]

    low_intensity_features = ["seg1_mean_png_intensity", "seg2_mean_png_intensity", "seg3_mean_png_intensity", "mean_png_intensity"]
    for feat in low_intensity_features:
        if feat in X.columns:
            X = X.drop(columns=[feat])
    
    X = X.dropna(axis=1, how='all')

    # Replace missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    loo = LeaveOneOut()
    model_preds = {name: [] for (name, _) in model_list}
    ensemble_preds = []
    actuals = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, model in model_list:
            model.fit(X_train, y_train)

        fold_preds = {}
        for name, model in model_list:
            pred = model.predict(X_test)
            model_preds[name].append(pred[0])
            fold_preds[name] = pred[0]

        ensemble_pred = np.mean([fold_preds[name] for name, _ in model_list])
        ensemble_preds.append(ensemble_pred)
        actuals.append(y_test.values[0])

    for name, _ in model_list:
        mse = mean_squared_error(actuals, model_preds[name])
        r2 = r2_score(actuals, model_preds[name])
        print(f"{name} Mean Squared Error: {mse}")
        print(f"{name} R² Score: {r2}")

    mse_ensemble = mean_squared_error(actuals, ensemble_preds)
    r2_ensemble = r2_score(actuals, ensemble_preds)
    print("---------------------------------")
    print(f"Ensemble Mean Squared Error: {mse_ensemble}")
    print(f"Ensemble R² Score: {r2_ensemble}")

# --------------------------
# 5) Plot Learning Curve
# --------------------------
def plot_learning_curve(df):
    from sklearn.base import BaseEstimator, RegressorMixin

    selected_models = []
    if use_rf:
        selected_models.append(RandomForestRegressor(n_estimators=100, random_state=42))
    if use_lr:
        selected_models.append(LinearRegression())
    if use_gb:
        selected_models.append(GradientBoostingRegressor(n_estimators=100, random_state=42))

    class EnsembleRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, models=None):
            self.models = models if models is not None else []

        def fit(self, X, y):
            for m in self.models:
                m.fit(X, y)
            return self

        def predict(self, X):
            if not self.models:
                return np.zeros(len(X))
            preds = [m.predict(X) for m in self.models]
            return np.mean(preds, axis=0)

    X = df.drop(columns=["airflow_rate"])
    y = df["airflow_rate"]

    X = X.dropna(axis=1, how='all')
    # Impute missing values in X using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    estimator = EnsembleRegressor(models=selected_models)

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv=5,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring='neg_mean_squared_error',
        random_state=42
    )
    train_scores_mean = -np.mean(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Error')
    plt.plot(train_sizes, validation_scores_mean, 'o-', label='Validation Error')
    plt.xlabel('Training Examples')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve for Ensemble Model')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    dataset_folder = os.path.join(os.getcwd(), "dataset_new")
    excel_path = os.path.join(os.getcwd(), "dataset_new", "DAQ_LW_Gypsum.xlsx")

    # Example ROI for IR-based gradient features (applied to IRVideoProcessing)
    roi = (100, 100, 200, 200)
    num_segments = 3

    df = process_dataset(dataset_folder, excel_path, num_segments=num_segments, roi=roi)
    if df.empty:
        print("No data processed. Check your dataset folder and Excel file.")
    else:
        print("\nExtracted Feature DataFrame:")
        print(df.head())
        train_and_evaluate(df)
        plot_learning_curve(df)
