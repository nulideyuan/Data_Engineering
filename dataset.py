# data/dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import os
import numpy as np
import matplotlib.pyplot as plt
import os, time, json, random


class NumpyDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(np.asarray(X)).float()
        if y is None:
            self.y = None
        else:
            self.y = torch.from_numpy(np.asarray(y)).float()

    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]





def load_mimic4_los(
    csv_path: str = "processed/mimic4_los.csv",
    target: str = "los_days",
    time_col: str = "admittime",
    split: tuple = (0.7, 0.1, 0.2),
    verbose: bool = True,
):
    """
    Load processed/mimic4_los.csv and return train/val/test splits ready for
    LOS regression and SHAP interpretability.

    Key guarantee: feature_cols is returned in the exact same order as the
    columns in X_train/X_val/X_test (continuous first, then binary), so you
    can pass it directly to shap.summary_plot as feature_names.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test,
    feature_cols,      # ordered list matching X columns — use for SHAP
    continuous_cols,   # subset that was RobustScaled
    binary_cols,       # subset that was NOT scaled
    scaler             # fitted RobustScaler (for inverse-transform if needed)
    """
    assert abs(sum(split) - 1.0) < 1e-6, "split ratios must sum to 1.0"

    df = pd.read_csv(csv_path, parse_dates=[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # ---- target ----
    y = df[target].to_numpy(dtype=np.float32)

    # ---- auto-detect feature groups ----
    # Columns that are clearly continuous by name pattern
    CONTINUOUS_EXACT = {
        "anchor_age",
        "admission_dayofweek",
        "admission_month",
        "admission_year",
    }
    skip = {target, time_col, "hadm_id"}

    continuous_cols = []
    binary_cols = []
    for c in df.columns:
        if c in skip:
            continue
        if (
            c in CONTINUOUS_EXACT
            or c.startswith("num_")
            or c.startswith("triage_")
        ):
            continuous_cols.append(c)
        else:
            binary_cols.append(c)

    # feature_cols order matches np.hstack([X_cont, X_bin]) — critical for SHAP
    feature_cols = continuous_cols + binary_cols

    X_cont = df[continuous_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    X_bin  = df[binary_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)

    # ---- temporal split ----
    N = len(df)
    i_tr = int(split[0] * N)
    i_va = int((split[0] + split[1]) * N)

    # ---- RobustScaler fitted on TRAIN continuous only ----
    scaler = RobustScaler()
    Xtr_cont = scaler.fit_transform(X_cont[:i_tr])
    Xva_cont = scaler.transform(X_cont[i_tr:i_va])
    Xte_cont = scaler.transform(X_cont[i_va:])

    Xtr_bin = X_bin[:i_tr]
    Xva_bin = X_bin[i_tr:i_va]
    Xte_bin = X_bin[i_va:]

    X_train = np.hstack([Xtr_cont, Xtr_bin]).astype(np.float32)
    X_val   = np.hstack([Xva_cont, Xva_bin]).astype(np.float32)
    X_test  = np.hstack([Xte_cont, Xte_bin]).astype(np.float32)

    y_train, y_val, y_test = y[:i_tr], y[i_tr:i_va], y[i_va:]

    if verbose:
        print(f"Total: {N:,} | TRAIN {X_train.shape} | VAL {X_val.shape} | TEST {X_test.shape}")
        print(f"Time  TRAIN: {df[time_col].iloc[0].date()} → {df[time_col].iloc[i_tr-1].date()}")
        print(f"      VAL  : {df[time_col].iloc[i_tr].date()} → {df[time_col].iloc[i_va-1].date()}")
        print(f"      TEST : {df[time_col].iloc[i_va].date()} → {df[time_col].iloc[-1].date()}")
        print(f"Features: {len(feature_cols)} total ({len(continuous_cols)} continuous, {len(binary_cols)} binary)")

    return (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        feature_cols,
        continuous_cols,
        binary_cols,
        scaler,
    )