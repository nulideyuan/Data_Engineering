"""
dataset.py — LOS dataset loaders for MIMIC-4, MIMIC-3, eICU, Synthea.

All loaders return the same 10-tuple:
    X_train, y_train, X_val, y_val, X_test, y_test,
    feature_cols,    # ordered list matching X columns — pass directly to SHAP
    continuous_cols, # RobustScaled
    binary_cols,     # not scaled (0/1 only)
    scaler           # fitted RobustScaler

Usage
-----
from dataset import load_mimic4_los, load_mimic3_los, load_eicu_los, load_synthea_los

X_train, y_train, X_val, y_val, X_test, y_test, \
feature_cols, continuous_cols, binary_cols, scaler = load_mimic4_los()
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler


# ── PyTorch dataset wrapper ──────────────────────────────────────────────────

class NumpyDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(np.asarray(X)).float()
        self.y = None if y is None else torch.from_numpy(np.asarray(y)).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i] if self.y is None else (self.X[i], self.y[i])


# ── Generic loader ───────────────────────────────────────────────────────────

def load_los_dataset(
    csv_path: str,
    target: str = "los_days",
    time_col: str = None,       # None → split on existing row order (eICU)
    split: tuple = (0.7, 0.1, 0.2),
    verbose: bool = True,
):
    """
    Generic LOS dataset loader. Works for MIMIC-4, MIMIC-3, eICU, Synthea.

    Continuous vs binary is detected from data:
      binary  = column whose non-null values are a subset of {0, 1}
      continuous = everything else

    RobustScaler is fitted on the training split of continuous columns only.
    feature_cols order = continuous_cols + binary_cols, matching X column order.
    """
    assert abs(sum(split) - 1.0) < 1e-6, "split ratios must sum to 1.0"

    df = pd.read_csv(csv_path)

    # Sort by time if available
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

    y = df[target].to_numpy(dtype=np.float32)

    # Columns to skip: target, time, and any ID-like columns
    ID_COLS = {"hadm_id", "patientunitstayid", "hospitalid",
               "encounter_id", "patient_id"}
    skip = {target}
    if time_col:
        skip.add(time_col)
    skip |= ID_COLS & set(df.columns)

    feature_raw = [c for c in df.columns if c not in skip]

    # Auto-detect binary vs continuous from actual values
    continuous_cols, binary_cols = [], []
    for c in feature_raw:
        uniq = set(df[c].dropna().unique())
        if uniq <= {0, 1, 0.0, 1.0}:
            binary_cols.append(c)
        else:
            continuous_cols.append(c)

    feature_cols = continuous_cols + binary_cols  # order matches X hstack

    X_cont = (df[continuous_cols]
              .apply(pd.to_numeric, errors="coerce")
              .fillna(0)
              .to_numpy(np.float32))
    X_bin  = (df[binary_cols]
              .apply(pd.to_numeric, errors="coerce")
              .fillna(0)
              .to_numpy(np.float32))

    # Temporal (or sequential) split
    N    = len(df)
    i_tr = int(split[0] * N)
    i_va = int((split[0] + split[1]) * N)

    scaler   = RobustScaler()
    Xtr_cont = scaler.fit_transform(X_cont[:i_tr])
    Xva_cont = scaler.transform(X_cont[i_tr:i_va])
    Xte_cont = scaler.transform(X_cont[i_va:])

    X_train = np.hstack([Xtr_cont, X_bin[:i_tr]]).astype(np.float32)
    X_val   = np.hstack([Xva_cont, X_bin[i_tr:i_va]]).astype(np.float32)
    X_test  = np.hstack([Xte_cont, X_bin[i_va:]]).astype(np.float32)

    y_train, y_val, y_test = y[:i_tr], y[i_tr:i_va], y[i_va:]

    if verbose:
        print(f"Dataset : {csv_path}")
        print(f"Total   : {N:,} | TRAIN {X_train.shape} | VAL {X_val.shape} | TEST {X_test.shape}")
        if time_col and time_col in df.columns:
            fmt = lambda i: str(df[time_col].iloc[i].date())
            print(f"Time  TRAIN: {fmt(0)} → {fmt(i_tr - 1)}")
            print(f"      VAL  : {fmt(i_tr)} → {fmt(i_va - 1)}")
            print(f"      TEST : {fmt(i_va)} → {fmt(-1)}")
        print(f"Features: {len(feature_cols)} total  "
              f"({len(continuous_cols)} continuous, {len(binary_cols)} binary)")

    return (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        feature_cols,
        continuous_cols,
        binary_cols,
        scaler,
    )


# ── Per-dataset convenience wrappers ─────────────────────────────────────────

def load_mimic4_los(csv_path: str = "processed/mimic4_los.csv", **kw):
    """MIMIC-4 hospital LOS. Time-based split on admittime."""
    return load_los_dataset(csv_path, target="los_days", time_col="admittime", **kw)


def load_mimic3_los(csv_path: str = "processed/mimic3_los.csv", **kw):
    """MIMIC-3 hospital LOS. Time-based split on admittime."""
    return load_los_dataset(csv_path, target="los_days", time_col="admittime", **kw)


def load_eicu_los(csv_path: str = "processed/eicu_los.csv", **kw):
    """eICU ICU LOS. No timestamp — split on existing row order."""
    return load_los_dataset(csv_path, target="icu_los_days", time_col=None, **kw)


def load_synthea_los(csv_path: str = "processed/synthea_los.csv", **kw):
    """Synthea synthetic inpatient LOS. Time-based split on admittime."""
    return load_los_dataset(csv_path, target="los_days", time_col="admittime", **kw)
