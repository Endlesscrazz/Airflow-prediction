#!/usr/bin/env python
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
from sklearn.impute import SimpleImputer  
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reload roi_detection if needed
importlib.reload(roi_detection)
from roi_detection import extract_roi_features

# --------------------------
# MODEL SELECTION FLAGS
# --------------------------
use_rf = True
use_lr = False
use_gb = True

# --------------------------
# ROI CONFIGURATIONS (imported from airflow_model)
# --------------------------
from airflow_model import ROI_CONFIGS, IRVideoProcessing, parse_delta_T, parse_airflow_rate

# --------------------------
# Process Dataset Function
# --------------------------
def process_dataset(dataset_folder, excel_path, num_segments=1, use_roi_features=True):
    """
    Process the dataset by iterating through subfolders (named like "FanPower_1.6V", etc.).
    Global features are always extracted using IRVideoProcessing.extract_features with roi=None.
    If use_roi_features is True, ROI-based features (from ROI_CONFIGS) are also extracted and merged.
    
    Args:
        dataset_folder (str): Root dataset folder.
        excel_path (str): Path to an Excel file for optional extra features.
        num_segments (int): Number of segments for global feature extraction.
        use_roi_features (bool): If True, add ROI-based features; if False, only global features are used.
    
    Returns:
        pd.DataFrame: Combined features DataFrame.
    """
    features_list = []
    # Iterate over subfolders (each should be named like "FanPower_1.6V", etc.)
    for folder_name in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        try:
            airflow_rate = parse_airflow_rate(folder_name)
        except ValueError as e:
            print(f"Skipping folder '{folder_name}' - {e}")
            continue

        # For ROI extraction, check if there is a config for this airflow_rate
        if use_roi_features and (airflow_rate not in ROI_CONFIGS):
            print(f"No ROI config found for airflow_rate={airflow_rate}, skipping ROI extraction in folder '{folder_name}'.")
            continue

        # List all .mat files in this folder
        mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
        if not mat_files:
            print(f"No .mat files found in folder: {folder_name}")
            continue

        for mat_file in mat_files:
            mat_filepath = os.path.join(folder_path, mat_file)
            try:
                delta_T = parse_delta_T(mat_file)

                # --- Global Features Extraction ---
                # Use global features extraction with roi=None (i.e. full frame)
                ir_video = IRVideoProcessing(mat_filepath, png_folder=folder_path)
                global_feats = ir_video.extract_features(num_segments=num_segments, roi=None)
                if global_feats is None:
                    print(f"Global feature extraction failed for {mat_file}")
                    continue

                combined_feats = {}
                combined_feats.update(global_feats)

                # --- ROI Features Extraction (if flag set) ---
                if use_roi_features:
                    cfg = ROI_CONFIGS[airflow_rate]
                    i_spot_center = np.array(cfg["i_spot_center"])
                    j_spot_center = np.array(cfg["j_spot_center"])
                    m_array_cfg = np.array(cfg["m_array"])
                    roi_feats = extract_roi_features(mat_filepath, i_spot_center, j_spot_center, m_array_cfg)
                    if roi_feats is None:
                        print(f"ROI feature extraction failed for {mat_file}")
                        continue
                    combined_feats.update(roi_feats)

                # --- Optional: Excel features extraction (if needed) ---
                sheet_name = f"{airflow_rate}V"
                if os.path.exists(excel_path):
                    try:
                        from airflow_model import extract_excel_features
                        excel_feats = extract_excel_features(excel_path, sheet_name)
                    except Exception as e:
                        print(f"Excel feature extraction error for {mat_file}: {e}")
                        excel_feats = {}
                else:
                    excel_feats = {}
                combined_feats.update(excel_feats)

                # Add metadata
                combined_feats["airflow_rate"] = airflow_rate
                combined_feats["delta_T"] = delta_T

                features_list.append(combined_feats)
            except Exception as e:
                print(f"Error processing {mat_file} in folder {folder_name}: {e}")
                continue

    return pd.DataFrame(features_list)

# --------------------------
# Train and Evaluate Function
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

    # Drop columns that are entirely missing.
    X = X.dropna(axis=1, how='all')

    loo = LeaveOneOut()
    model_preds = {name: [] for (name, _) in model_list}
    ensemble_preds = []
    actuals = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_preds = {}
        for name, model in model_list:
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            model_preds[name].append(pred[0])
            fold_preds[name] = pred[0]

        ensemble_pred = np.mean(list(fold_preds.values()))
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
# Plot Learning Curve Function
# --------------------------
def plot_learning_curve(df):
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    selected_models = []
    if use_rf:
        selected_models.append(RandomForestRegressor(n_estimators=100, random_state=42))
    if use_lr:
        selected_models.append(LinearRegression())
    if use_gb:
        selected_models.append(GradientBoostingRegressor(n_estimators=100, random_state=42))

    pipelines = []
    for model in selected_models:
        pipelines.append(Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('model', model)
        ]))

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

    estimator = EnsembleRegressor(models=pipelines)
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
    plt.title('Learning Curve for Ensemble Model with PCA')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    dataset_folder = os.path.join(os.getcwd(), "dataset_new")
    excel_path = os.path.join(os.getcwd(), "dataset_new", "DAQ_LW_Gypsum.xlsx")

    num_segments = 3

    print("----- Running with ROI features -----")
    df_with_roi = process_dataset(dataset_folder, excel_path, num_segments=num_segments, use_roi_features=True)
    if df_with_roi.empty:
        print("No data processed (with ROI). Check your dataset folder and Excel file.")
    else:
        print("\nExtracted Feature DataFrame (with ROI):")
        print(df_with_roi.head())
        train_and_evaluate(df_with_roi)
        plot_learning_curve(df_with_roi)

    print("\n----- Running without ROI features -----")
    df_without_roi = process_dataset(dataset_folder, excel_path, num_segments=num_segments, use_roi_features=False)
    if df_without_roi.empty:
        print("No data processed (without ROI). Check your dataset folder and Excel file.")
    else:
        print("\nExtracted Feature DataFrame (without ROI):")
        print(df_without_roi.head())
        train_and_evaluate(df_without_roi)
        plot_learning_curve(df_without_roi)
