"""
eICU LOS Dataset Builder (via BigQuery)
========================================
Produces processed/eicu_los.csv — one row per ICU stay, with:
  - icu_los_days          : target (ICU length of stay in days)
  - demographics          : age, gender, BMI, ethnicity
  - admission context     : ICU type, admit source, time features
  - APACHE physiology     : 20 continuous variables at ICU admission
  - APACHE comorbidities  : 10 binary flags (replaces Elixhauser)
  - APACHE score          : overall severity score
  - count features        : num_labs, num_medications

Run:
    source ~/my_python_env/venv/bin/activate
    cd ~/VS_Studio/Data_Engineering
    python build_eicu_los_bigquery.py
"""

import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "eicu-494316"
DATASET    = "physionet-data.eicu_crd"
OUT        = "processed"
os.makedirs(OUT, exist_ok=True)

client = bigquery.Client(project=PROJECT_ID)

def bq(sql: str, label: str = "") -> pd.DataFrame:
    if label:
        print(f"  → {label} ...")
    return client.query(sql).to_dataframe()


# ============================================================
# 1. Patient — core table, LOS target, demographics
# ============================================================
print("1. Patient ...")
pat = bq(f"""
SELECT
  patientunitstayid,
  patienthealthsystemstayid,
  hospitalid,
  uniquepid,
  gender,
  age,
  ethnicity,
  admissionweight,
  admissionheight,
  unittype,
  hospitaladmitsource,
  unitadmitsource,
  unitdischargestatus,
  hospitaldischargestatus,
  unitdischargeoffset,
  hospitaldischargeoffset,
  hospitaladmitoffset
FROM `{DATASET}.patient`
""", "patient")

# --- LOS target (ICU stay, in days) ---
pat["icu_los_days"] = pat["unitdischargeoffset"] / 1440.0

# Filter: valid ICU LOS
pat = pat[(pat["icu_los_days"] > 0) & (pat["icu_los_days"] <= 90)].copy()
print(f"   {len(pat):,} ICU stays after filter")

# --- Age: stored as string, "> 89" → 89 ---
pat["anchor_age"] = (
    pat["age"]
    .astype(str)
    .str.replace("> ", "", regex=False)
    .str.strip()
    .pipe(pd.to_numeric, errors="coerce")
    .clip(0, 89)
)

# --- Gender ---
pat["gender_bin"] = (
    pat["gender"].str.upper()
    .map({"FEMALE": 0, "MALE": 1})
    .fillna(0).astype(np.int8)
)

# --- Ethnicity (label encode, sorted by frequency) ---
eth_order = pat["ethnicity"].value_counts().index.tolist()
eth_map   = {e: i for i, e in enumerate(eth_order)}
pat["ethnicity_cat"] = pat["ethnicity"].map(eth_map).fillna(len(eth_order)).astype(np.int8)

# --- BMI ---
pat["bmi"] = (
    pat["admissionweight"] /
    ((pat["admissionheight"] / 100) ** 2)
).clip(10, 80)

# --- ICU type (label encode) ---
unit_order = pat["unittype"].value_counts().index.tolist()
unit_map   = {u: i for i, u in enumerate(unit_order)}
pat["unittype_cat"] = pat["unittype"].map(unit_map).fillna(len(unit_order)).astype(np.int8)

# --- Admit source (label encode) ---
src_order = pat["unitadmitsource"].value_counts().index.tolist()
src_map   = {s: i for i, s in enumerate(src_order)}
pat["unitadmitsource_cat"] = pat["unitadmitsource"].map(src_map).fillna(len(src_order)).astype(np.int8)

# --- In-hospital death flag ---
pat["hospital_expire_flag"] = (
    pat["hospitaldischargestatus"].str.upper()
    .eq("EXPIRED").astype(np.int8)
)


# ============================================================
# 2. APACHE APS Variables — physiology at ICU admission
# ============================================================
print("2. APACHE APS variables ...")
aps = bq(f"""
SELECT
  patientunitstayid,
  intubated,
  vent,
  dialysis,
  eyes,
  motor,
  verbal,
  heartrate,
  meanbp,
  temperature,
  respiratoryrate,
  wbc,
  hematocrit,
  sodium,
  ph,
  pao2,
  pco2,
  fio2,
  creatinine,
  bilirubin,
  albumin,
  glucose,
  bun,
  urine
FROM `{DATASET}.apacheapsvar`
""", "apacheapsvar")

# Rename for clarity
aps = aps.rename(columns={
    "intubated":       "apache_intubated",
    "vent":            "apache_vent",
    "dialysis":        "apache_dialysis",
    "eyes":            "apache_gcs_eyes",
    "motor":           "apache_gcs_motor",
    "verbal":          "apache_gcs_verbal",
    "heartrate":       "apache_heartrate",
    "meanbp":          "apache_meanbp",
    "temperature":     "apache_temperature",
    "respiratoryrate": "apache_resprate",
    "wbc":             "apache_wbc",
    "hematocrit":      "apache_hematocrit",
    "sodium":          "apache_sodium",
    "ph":              "apache_ph",
    "pao2":            "apache_pao2",
    "pco2":            "apache_pco2",
    "fio2":            "apache_fio2",
    "creatinine":      "apache_creatinine",
    "bilirubin":       "apache_bilirubin",
    "albumin":         "apache_albumin",
    "glucose":         "apache_glucose",
    "bun":             "apache_bun",
    "urine":           "apache_urine",
})


# ============================================================
# 3. APACHE Predictor Variables — comorbidities
# ============================================================
print("3. APACHE predictor variables (comorbidities) ...")
pred = bq(f"""
SELECT
  patientunitstayid,
  diabetes,
  immunosuppression,
  hepaticfailure,
  cirrhosis,
  leukemia,
  metastaticcancer,
  aids,
  lymphoma,
  electivesurgery,
  readmit
FROM `{DATASET}.apachepredvar`
""", "apachepredvar")

# Rename with cm_ prefix to match MIMIC style
pred = pred.rename(columns={
    "diabetes":         "cm_diabetes",
    "immunosuppression":"cm_immunosuppression",
    "hepaticfailure":   "cm_hepatic_failure",
    "cirrhosis":        "cm_cirrhosis",
    "leukemia":         "cm_leukemia",
    "metastaticcancer": "cm_metastatic",
    "aids":             "cm_aids",
    "lymphoma":         "cm_lymphoma",
    "electivesurgery":  "cm_elective_surgery",
    "readmit":          "cm_readmit",
})

cm_cols = [c for c in pred.columns if c.startswith("cm_")]
for c in cm_cols:
    pred[c] = pd.to_numeric(pred[c], errors="coerce").fillna(0).clip(0, 1).astype(np.int8)


# ============================================================
# 4. APACHE Patient Result — overall severity score
# ============================================================
print("4. APACHE patient result ...")
apache = bq(f"""
SELECT
  patientunitstayid,
  apachescore,
  predictedhospitalmortality
FROM `{DATASET}.apachepatientresult`
WHERE apacheversion IN ('IVa', 'IVb')
  AND apachescore IS NOT NULL
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY patientunitstayid
    ORDER BY apachescore DESC
) = 1
""", "apachepatientresult")

apache = apache.rename(columns={
    "apachescore":                "apache_score",
    "predictedhospitalmortality": "apache_pred_mortality",
})


# ============================================================
# 5. Lab count
# ============================================================
print("5. Lab count ...")
num_labs = bq(f"""
SELECT patientunitstayid, COUNT(*) AS num_labs
FROM `{DATASET}.lab`
GROUP BY patientunitstayid
""", "lab")


# ============================================================
# 6. Medication count
# ============================================================
print("6. Medication count ...")
num_meds = bq(f"""
SELECT patientunitstayid, COUNT(*) AS num_medications
FROM `{DATASET}.medication`
GROUP BY patientunitstayid
""", "medication")


# ============================================================
# 7. Merge
# ============================================================
print("7. Merging ...")

keep_cols = [
    "patientunitstayid", "hospitalid",
    "icu_los_days",
    "anchor_age", "gender_bin", "ethnicity_cat", "bmi",
    "unittype_cat", "unitadmitsource_cat",
    "hospital_expire_flag",
    "unitdischargeoffset",   # keep for reference / time sort
]
df = pat[keep_cols].copy()

for tbl in [aps, pred, apache, num_labs, num_meds]:
    df = df.merge(tbl, on="patientunitstayid", how="left")

# Fill counts
for c in ["num_labs", "num_medications"]:
    df[c] = df[c].fillna(0).astype(int)

# Fill comorbidities
cm_cols = [c for c in df.columns if c.startswith("cm_")]
df[cm_cols] = df[cm_cols].fillna(0).astype(np.int8)

# APACHE physiology: -1 is eICU sentinel for "not measured" → NaN
# Keep as NaN — LightGBM/XGBoost handle missing natively; filling 0 is wrong
# (apache_meanbp=0 means no blood pressure, not "not measured")
apache_binary = ["apache_intubated", "apache_vent", "apache_dialysis"]
apache_cont   = [c for c in df.columns if c.startswith("apache_") and c not in apache_binary]
df[apache_cont] = (
    df[apache_cont]
    .apply(pd.to_numeric, errors="coerce")
    .replace(-1, np.nan)
)
df["apache_urine"] = df["apache_urine"].clip(lower=0)   # corrupt large negatives
# Binary flags: -1 → 0 (not applicable = not used)
df[apache_binary] = (
    df[apache_binary]
    .apply(pd.to_numeric, errors="coerce")
    .replace(-1, 0)
    .fillna(0)
    .astype(np.int8)
)

# Sort by unitdischargeoffset as proxy for time (no absolute timestamp in eICU)
df = df.sort_values("unitdischargeoffset").reset_index(drop=True)
df = df.drop(columns=["unitdischargeoffset"])

# ============================================================
# 8. Save
# ============================================================
out_path = f"{OUT}/eicu_los.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

feature_cols = [
    c for c in df.columns
    if c not in ("patientunitstayid", "hospitalid", "icu_los_days")
]
apache_cont = [c for c in feature_cols if c.startswith("apache_") and c not in
               ["apache_intubated","apache_vent","apache_dialysis"]]
binary_cols = [c for c in feature_cols if c not in apache_cont]

print(f"\nTarget  : icu_los_days  mean={df['icu_los_days'].mean():.2f}  median={df['icu_los_days'].median():.2f}")
print(f"Features: {len(feature_cols)} total")
print(f"  APACHE physiology (continuous): {len(apache_cont)}")
print(f"  Binary/categorical            : {len(binary_cols)}")
print(f"Missing : {df[feature_cols].isna().sum().sum()}")
print(f"\nComorbidity prevalence:")
for c in cm_cols:
    if c in df.columns:
        print(f"  {c:<28} {df[c].mean():.1%}")
