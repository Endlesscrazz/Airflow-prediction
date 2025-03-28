import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

import importlib
import roi_detection
from roi_detection import extract_roi_features

importlib.reload(roi_detection)

# Flags to select models
use_rf = True
use_lr = False
use_gb = True
use_roi_features = True

# --------------------------
# Helper Functions
# --------------------------
def parse_airflow_rate(foldername):
    """
    Extract the airflow rate label from the folder name.
    Example: "FanPower_1.6V" returns 1.6.
    """
    match = re.search(r'FanPower_(\d+(\.\d+)?)V', foldername, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not parse airflow rate in folder name: {foldername}")

def parse_delta_T(filename):
    """
    Extract delta T from the MATLAB file name.
    Example: "temp_2025-3-7-19-45-8_21.4_35_13.6_mat" returns 13.6.
    """
    tokens = filename.split('_')
    try:
        delta_T = float(tokens[-2])
        return delta_T
    except Exception as e:
        raise ValueError(f"Could not parse delta T from filename {filename}: {e}")

# --------------------------
# IR Video Processing Class
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
            mat_data = scipy.io.loadmat(self.mat_filepath, squeeze_me=True)
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
            frames = [plt.imread(os.path.join(self.png_folder, f)) for f in png_files]
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
                frame_max = np.max(data_used, axis=(1,2))
                frame_mean = np.mean(data_used, axis=(1,2))
                frame_std = np.std(data_used, axis=(1,2))
                features.update({
                    "overall_max": np.max(frame_max),
                    "overall_mean": np.mean(frame_mean),
                    "overall_std": np.mean(frame_std)
                })
                spatial_grad = self.compute_spatial_gradient(roi=roi)
                avg_spatial = np.mean(spatial_grad, axis=(1,2))
                features.update({
                    "mean_spatial_grad": np.mean(avg_spatial),
                    "std_spatial_grad": np.std(avg_spatial)
                })
                temporal_grad = self.compute_temporal_gradient(roi=roi)
                valid_temp = temporal_grad[:-1]
                avg_temp = np.mean(valid_temp, axis=(1,2))
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
                    end = num_frames if seg == num_segments-1 else (seg+1)*segment_length
                    seg_data = data_used[start:end]
                    seg_max = np.max(seg_data, axis=(1,2))
                    seg_mean = np.mean(seg_data, axis=(1,2))
                    seg_std = np.std(seg_data, axis=(1,2))
                    features[f"seg{seg+1}_overall_max"] = np.max(seg_max)
                    features[f"seg{seg+1}_overall_mean"] = np.mean(seg_mean)
                    features[f"seg{seg+1}_overall_std"] = np.mean(seg_std)
                    seg_spatial = []
                    for frame in seg_data:
                        gx = np.gradient(frame, axis=0)
                        gy = np.gradient(frame, axis=1)
                        seg_spatial.append(np.sqrt(gx**2 + gy**2))
                    seg_spatial = np.array(seg_spatial)
                    seg_avg_spatial = np.mean(seg_spatial, axis=(1,2))
                    features[f"seg{seg+1}_mean_spatial_grad"] = np.mean(seg_avg_spatial)
                    features[f"seg{seg+1}_std_spatial_grad"] = np.std(seg_avg_spatial)
                    seg_temp = np.zeros_like(seg_data)
                    if seg_data.shape[0] > 1:
                        seg_temp[:-1] = np.abs(seg_data[1:] - seg_data[:-1])
                        seg_valid_temp = seg_temp[:-1]
                        seg_avg_temp = np.mean(seg_valid_temp, axis=(1,2))
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
# Excel Features Extraction
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
# Dataset Processing
# --------------------------
def process_dataset(dataset_folder, excel_path, num_segments=1, roi=None,
                    i_spot_center=None, j_spot_center=None, m_array=None):
    """
    Process each subfolder in the dataset. For each folder, extract the airflow rate,
    look up the ROI parameters based on a predefined mapping, and extract features.
    """
    roi_configs = {
        1.6: {"i_spot_center": [340, 300, 400],
              "j_spot_center": [305, 300, 400],
              "m_array": [2, 4, 4]},
        1.8: {"i_spot_center": [340, 300, 400],
              "j_spot_center": [305, 300, 400],
              "m_array": [2, 4, 4]},
        2.1: {"i_spot_center": [340, 300, 400],
              "j_spot_center": [305, 300, 400],
              "m_array": [2, 4, 4]},
        2.4: {"i_spot_center": [311, 300, 400],
              "j_spot_center": [252, 300, 400],
              "m_array": [2, 4, 4]}
    }
    features_list = []
    for item in os.listdir(dataset_folder):
        item_path = os.path.join(dataset_folder, item)
        if os.path.isdir(item_path):
            folder_name = item
            try:
                airflow_rate = parse_airflow_rate(folder_name)
            except Exception as e:
                print(f"Skipping folder {folder_name}: {e}")
                continue
            roi_params = roi_configs.get(airflow_rate, None)
            if roi_params is None:
                print(f"No ROI configuration for airflow rate {airflow_rate}; skipping folder {folder_name}")
                continue
            mat_files = [f for f in os.listdir(item_path) if f.endswith(".mat")]
            if not mat_files:
                print(f"No .mat file found in folder: {item}")
                continue
            for mat_file in mat_files:
                print(f"Processing folder: {folder_name}, file: {mat_file}")
                mat_filepath = os.path.join(item_path, mat_file)
                try:
                    delta_T = parse_delta_T(mat_file)
                    print(f"Delta T: {delta_T}")
                    png_folder = item_path  # PNG frames in same folder
                    ir_video = IRVideoProcessing(mat_filepath, png_folder)
                    ir_roi = roi if roi else (100, 100, 200, 200)
                    ir_features = ir_video.extract_features(num_segments=num_segments, roi=ir_roi)
                    sheet_name = f"{airflow_rate}V"
                    if os.path.exists(excel_path):
                        excel_features = extract_excel_features(excel_path, sheet_name)
                        if excel_features is None:
                            excel_features = {}
                    else:
                        excel_features = {}
                    if use_roi_features:
                        roi_features = extract_roi_features(mat_filepath,
                                                            roi_params["i_spot_center"],
                                                            roi_params["j_spot_center"],
                                                            roi_params["m_array"])
                        if roi_features is None:
                            roi_features = {}
                    else:
                        roi_features = {}
                    combined_features = {**ir_features, **excel_features, **roi_features,
                                         "airflow_rate": airflow_rate, "delta_T": delta_T}
                    features_list.append(combined_features)
                except Exception as e:
                    print(f"Error processing {folder_name}: {e}")
                    continue
        elif os.path.isfile(item_path) and item.endswith(".mat"):
            try:
                airflow_rate = parse_airflow_rate(os.path.splitext(item)[0])
                roi_params = roi_configs.get(airflow_rate, None)
                if roi_params is None:
                    print(f"No ROI configuration for airflow rate {airflow_rate}; skipping file {item}")
                    continue
                print(f"Processing file: {item}")
                delta_T = parse_delta_T(item)
                mat_filepath = item_path
                png_folder = dataset_folder
                ir_video = IRVideoProcessing(mat_filepath, png_folder)
                ir_roi = roi if roi else (100, 100, 200, 200)
                ir_features = ir_video.extract_features(num_segments=num_segments, roi=ir_roi)
                sheet_name = f"{airflow_rate}V"
                if os.path.exists(excel_path):
                    excel_features = extract_excel_features(excel_path, sheet_name)
                    if excel_features is None:
                        excel_features = {}
                else:
                    excel_features = {}
                if use_roi_features:
                    roi_features = extract_roi_features(mat_filepath,
                                                        roi_params["i_spot_center"],
                                                        roi_params["j_spot_center"],
                                                        roi_params["m_array"])
                    if roi_features is None:
                        roi_features = {}
                else:
                    roi_features = {}
                combined_features = {**ir_features, **excel_features, **roi_features,
                                     "airflow_rate": airflow_rate, "delta_T": delta_T}
                features_list.append(combined_features)
            except Exception as e:
                print(f"Error processing file {item}: {e}")
                continue
    return pd.DataFrame(features_list)

# --------------------------
# Model Training & Evaluation with PCA
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

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    # --- Apply PCA ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    print("Original number of features:", X.shape[1])
    print("Reduced number of features after PCA:", X_pca.shape[1])

    loo = LeaveOneOut()
    model_preds = {name: [] for (name, _) in model_list}
    ensemble_preds = []
    actuals = []
    
    for train_idx, test_idx in loo.split(X_pca):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
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
# Learning Curve Plotting (with PCA)
# --------------------------
def plot_learning_curve(df):
    from sklearn.impute import SimpleImputer

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
    
    low_intensity_features = ["seg1_mean_png_intensity", "seg2_mean_png_intensity", "seg3_mean_png_intensity", "mean_png_intensity"]
    for feat in low_intensity_features:
        if feat in X.columns:
            X = X.drop(columns=[feat])
    X = X.dropna(axis=1, how='all')
    
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    # --- Apply PCA for learning curve ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    estimator = EnsembleRegressor(models=selected_models)
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X_pca, y, cv=5,
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
    plt.title('Learning Curve for Ensemble Model with PCA')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# --------------------------
# Exploratory Data Analysis (EDA)
# --------------------------
def visualize_sample_frame(dataset_folder):
    mat_files = [f for f in os.listdir(dataset_folder) if f.endswith(".mat")]
    if not mat_files:
        print("No .mat files found in dataset folder.")
        return
    sample_file = os.path.join(dataset_folder, mat_files[0])
    mat_data = scipy.io.loadmat(sample_file, squeeze_me=True)
    frames = mat_data["big_array"]
    if frames.ndim == 1 or (frames.dtype == object):
        frames = np.stack([np.asarray(frame) for frame in frames])
    elif frames.ndim == 2:
        frames = frames[np.newaxis, :, :]
    print("Sample .mat file:", sample_file)
    print("Frame shape (num_frames, height, width):", frames.shape)
    plt.figure(figsize=(6,5))
    plt.imshow(frames[0], cmap='hot')
    plt.title("Sample IR Frame (First Frame)")
    plt.colorbar(label="Normalized Intensity")
    plt.show()
    print("Frame Statistics:")
    print("Mean:", np.mean(frames[0]))
    print("Std:", np.std(frames[0]))
    print("Min:", np.min(frames[0]), "Max:", np.max(frames[0]))

def load_global_features():
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
            png_folder = item_path if os.path.isdir(item_path) else dataset_folder
            ir_video = IRVideoProcessing(mat_filepath, png_folder)
            global_features = ir_video.extract_features(num_segments=num_segments, roi=roi)
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

# --------------------------
# Main Execution
# --------------------------
dataset_folder = os.path.join(os.getcwd(), "dataset_new")
excel_path = os.path.join(os.getcwd(), "dataset_new", "DAQ_LW_Gypsum.xlsx")

# ROI and feature extraction parameters
roi = (100, 100, 200, 200)
i_spot_center = np.array([329, 331, 524])
j_spot_center = np.array([325, 400, 413])
m_array = np.array([2, 2, 2])
num_segments = 1

# Visualize a sample IR frame for EDA
visualize_sample_frame(dataset_folder)

# Create combined feature DataFrame using the updated process_dataset function
combined_df = process_dataset(dataset_folder, excel_path, num_segments=num_segments, roi=roi,
                              i_spot_center=i_spot_center, j_spot_center=j_spot_center, m_array=m_array)
print("\nCombined Features DataFrame shape:", combined_df.shape)
print("Combined Features Columns:\n", combined_df.columns.tolist())

global_df = load_global_features()
roi_df = load_roi_features()
print("\nGlobal Features DataFrame shape:", global_df.shape)
print("ROI Features DataFrame shape:", roi_df.shape)

# Correlation Analysis
corr_matrix = combined_df.drop(columns=["airflow_rate"]).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size":8}, square=True)
plt.title("Correlation Matrix (Lower Triangle) of Combined Features")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

target_corr = combined_df.corr()["airflow_rate"].drop("airflow_rate")
print("\nCorrelation of each feature with airflow_rate:")
print(target_corr.sort_values(ascending=False))

sns.clustermap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", annot_kws={"size":8})
plt.title("Clustered Correlation Matrix of Combined Features")
plt.show()

# Feature Importance using RandomForest
X = combined_df.drop(columns=["airflow_rate"])
y = combined_df["airflow_rate"]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):")
print(feat_imp)
plt.figure(figsize=(10,6))
feat_imp.plot(kind='bar')
plt.title("Feature Importances from RandomForestRegressor")
plt.ylabel("Importance")
plt.show()

# Model Evaluation & Ablation Studies
print("\n--- Evaluating Combined Features ---")
train_and_evaluate(combined_df)
print("\n--- Evaluating Global Features Only ---")
train_and_evaluate(global_df)
print("\n--- Evaluating ROI Features Only ---")
train_and_evaluate(roi_df)

# Plot Learning Curve
plot_learning_curve(combined_df)
