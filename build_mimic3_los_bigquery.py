"""
MIMIC-3 LOS Dataset Builder (via BigQuery)
===========================================
Produces processed/mimic3_los.csv — same structure as mimic4_los.csv but:
  - Age calculated from DOB (capped at 89, MIMIC-3 shifts >89 yr DOBs)
  - ICD-9 only (no ICD-10) for Elixhauser comorbidities
  - No ED triage vitals (MIMIC-3 has no triage table)
  - has_icu from ICUSTAYS table

Run from project root:
    source ~/my_python_env/venv/bin/activate
    python build_mimic3_los_bigquery.py
"""

import os
import re
import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "eicu-494316"
DATASET    = "physionet-data.mimiciii_clinical"
OUT        = "processed"
os.makedirs(OUT, exist_ok=True)

client = bigquery.Client(project=PROJECT_ID)

def bq(sql: str) -> pd.DataFrame:
    print(f"  → querying {len(sql)} chars ...")
    return client.query(sql).to_dataframe()


# ============================================================
# Elixhauser ICD-9 patterns only
# ============================================================
elixhauser_icd9 = {
    "chf": [
        "39891","40201","40211","40291",
        "40401","40403","40411","40413","40491","40493",
        "4254","4255","4256","4257","4258","4259","428",
    ],
    "arrhythmia": ["426","427","7850","99601","99604","V450","V533"],
    "valve": [
        "0932","394","395","396","397","424",
        "7463","7464","7465","7466","V422","V433",
    ],
    "pulm_circ": ["4150","4151","416","417"],
    "pvd": [
        "0930","4373","440","441","442",
        "4431","4432","4433","4434","4435","4436","4437","4438","4439",
        "4471","5571","5579","V434",
    ],
    "htn_uncomp": ["4011","4019","6420"],
    "htn_comp":   ["4010","402","403","404","405","6421","6422","6427","6429"],
    "paralysis":  ["3341","342","343","344","4382","4383","4384","4385"],
    "neuro_other": [
        "330","331","332","3334","3335","33392",
        "334","335","3362","340","341","345","347","3481","3483","7803","7843",
    ],
    "copd": [
        "490","491","492","493","494","495","496","497","498","499",
        "500","501","502","503","504","505","5064","5081","5088",
    ],
    "dm_uncomp":    ["2500","2501","2502","2503","6480"],
    "dm_comp":      ["2504","2505","2506","2507","2508","2509","7751"],
    "hypothyroid":  ["2409","243","244","2461","2468"],
    "renal": [
        "40301","40311","40391",
        "40402","40403","40412","40413","40492","40493",
        "585","586","5880","V420","V451","V56",
    ],
    "liver": [
        "07022","07023","07032","07033","07044","07054","0706","0709",
        "4560","4561","4562","570","571",
        "5722","5723","5724","5725","5726","5727","5728",
        "5733","5734","5738","5739","V427",
    ],
    "ulcer": ["5317","5319","5327","5329","5337","5339","5347","5349","V1271"],
    "hiv":   ["042","043","044"],
    "lymphoma": ["200","201","202","2030","2386","2733","V1071","V1072","V1079"],
    "metastatic":  ["196","197","198","199"],
    "solid_tumor": [
        "140","141","142","143","144","145","146","147","148","149",
        "150","151","152","153","154","155","156","157","158","159",
        "160","161","162","163","164","165","166","167","168","169",
        "170","171","172",
        "174","175","179","180","181","182","183","184","185","186","187","188","189",
        "190","191","192","193","194","195","V10",
    ],
    "rheumatoid": [
        "446","7010","710","7112","714","7193","720","725",
        "7285","72889","72930",
    ],
    "coagulopathy": ["286","2871","2873","2874","2875"],
    "obesity":      ["2780"],
    "weight_loss":  ["260","261","262","263","7832","7994"],
    "fluid_electrolyte": ["2536","276"],
    "blood_loss":   ["2800","6482"],
    "anemia":       ["2801","2802","2803","2804","2805","2806","2807","2808","2809","281","2852","2859"],
    "alcohol": [
        "2652","291","303","3050","3575","4255","5353",
        "5710","5711","5712","5713","980","V113",
    ],
    "drug":     ["292","304","3052","3053","3054","3055","3056","3057","3058","3059","V6542"],
    "psychosis":["2938","295","29604","29614","29644","29654","297","298"],
    "depression":["2962","2963","2965","3004","30112","309","311"],
}

def _build_regex(codes):
    if not codes:
        return None
    escaped = sorted(set(re.escape(c) for c in codes), key=len, reverse=True)
    return re.compile("^(?:" + "|".join(escaped) + ")")

icd9_regex = {cm: _build_regex(codes) for cm, codes in elixhauser_icd9.items()}


def compute_elixhauser(diag: pd.DataFrame) -> pd.DataFrame:
    """diag must have columns: hadm_id, icd9_code (str)"""
    all_hadm = diag["hadm_id"].unique()
    cm_df = pd.DataFrame({"hadm_id": all_hadm})
    for cm, rx in icd9_regex.items():
        if rx is None:
            cm_df[f"cm_{cm}"] = np.int8(0)
            continue
        matched = set(diag.loc[diag["icd9_code"].str.match(rx, na=False), "hadm_id"])
        cm_df[f"cm_{cm}"] = cm_df["hadm_id"].isin(matched).astype(np.int8)
    return cm_df


# ============================================================
# 1. Admissions + LOS
# ============================================================
print("1. Admissions ...")
adm = bq(f"""
SELECT
  subject_id, hadm_id,
  admittime, dischtime,
  admission_type,
  admission_location,
  insurance,
  hospital_expire_flag
FROM `{DATASET}.admissions`
""")
adm["admittime"] = pd.to_datetime(adm["admittime"])
adm["dischtime"] = pd.to_datetime(adm["dischtime"])
adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400.0
adm = adm[(adm["los_days"] > 0) & (adm["los_days"] <= 365)].copy()
print(f"   {len(adm):,} admissions after filter")

adm["admission_dayofweek"] = adm["admittime"].dt.weekday
adm["admission_month"]     = adm["admittime"].dt.month
adm["admission_year"]      = adm["admittime"].dt.year
adm["is_weekend"]          = adm["admittime"].dt.weekday.isin([5, 6]).astype(np.int8)

emergency_types = {"EMERGENCY", "URGENT"}
adm["is_emergency"] = adm["admission_type"].str.upper().isin(emergency_types).astype(np.int8)

ins_map = {"Medicare": 0, "Medicaid": 1, "Private": 2, "Government": 2, "Self Pay": 3}
adm["insurance_cat"] = adm["insurance"].map(ins_map).fillna(3).astype(np.int8)

loc_order = adm["admission_location"].value_counts().index.tolist()
loc_map   = {l: i for i, l in enumerate(loc_order)}
adm["admission_location_cat"] = adm["admission_location"].map(loc_map).fillna(len(loc_order)).astype(np.int8)

# ============================================================
# 2. Patients — age from DOB
# ============================================================
print("2. Patients ...")
pat = bq(f"""
SELECT subject_id, gender, dob
FROM `{DATASET}.patients`
""")
pat["dob"] = pd.to_datetime(pat["dob"])
pat["gender_bin"] = pat["gender"].str.upper().map({"F": 0, "M": 1}).fillna(0).astype(np.int8)

adm = adm.merge(pat[["subject_id", "gender_bin", "dob"]], on="subject_id", how="left")
adm["anchor_age"] = ((adm["admittime"] - adm["dob"]).dt.days / 365.25).clip(0, 89).round(1)
adm = adm.drop(columns=["dob"])

# ============================================================
# 3. Diagnoses → Elixhauser + count
# ============================================================
print("3. Diagnoses ICD ...")
diag = bq(f"""
SELECT hadm_id, icd9_code
FROM `{DATASET}.diagnoses_icd`
""")
diag["icd9_code"] = diag["icd9_code"].astype(str).str.strip()

num_diagnoses = diag.groupby("hadm_id").size().rename("num_diagnoses").reset_index()

print("   Computing Elixhauser comorbidities ...")
cm_df = compute_elixhauser(diag)
del diag

# ============================================================
# 4. Procedures count
# ============================================================
print("4. Procedures ...")
num_procedures = bq(f"""
SELECT hadm_id, COUNT(*) AS num_procedures
FROM `{DATASET}.procedures_icd`
GROUP BY hadm_id
""")

# ============================================================
# 5. Transfers count + ICU flag
# ============================================================
print("5. Transfers ...")
num_transfers = bq(f"""
SELECT hadm_id, COUNT(*) AS num_transfers
FROM `{DATASET}.transfers`
GROUP BY hadm_id
""")

has_icu = bq(f"""
SELECT DISTINCT hadm_id, 1 AS has_icu
FROM `{DATASET}.icustays`
""")
has_icu["has_icu"] = has_icu["has_icu"].astype(np.int8)

# ============================================================
# 6. Prescriptions count
# ============================================================
print("6. Prescriptions ...")
num_prescriptions = bq(f"""
SELECT hadm_id, COUNT(*) AS num_prescriptions
FROM `{DATASET}.prescriptions`
GROUP BY hadm_id
""")

# ============================================================
# 7. Lab events count
# ============================================================
print("7. Lab events ...")
num_labs = bq(f"""
SELECT hadm_id, COUNT(*) AS num_labs
FROM `{DATASET}.labevents`
WHERE hadm_id IS NOT NULL
GROUP BY hadm_id
""")

# ============================================================
# 8. Merge
# ============================================================
print("8. Merging ...")
df = adm.drop(columns=["admission_type", "admission_location",
                        "insurance", "dischtime", "subject_id"], errors="ignore")

for tbl in [num_diagnoses, cm_df, num_procedures, num_transfers,
            has_icu, num_prescriptions, num_labs]:
    df = df.merge(tbl, on="hadm_id", how="left")

# Fill counts
count_cols = ["num_diagnoses", "num_procedures", "num_transfers",
              "num_prescriptions", "num_labs"]
df[count_cols] = df[count_cols].fillna(0).astype(int)
df["has_icu"] = df["has_icu"].fillna(0).astype(np.int8)

cm_cols = [c for c in df.columns if c.startswith("cm_")]
df[cm_cols] = df[cm_cols].fillna(0).astype(np.int8)

df = df.sort_values("admittime").reset_index(drop=True)

# ============================================================
# 9. Save
# ============================================================
out_path = f"{OUT}/mimic3_los.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

feature_cols = [c for c in df.columns if c not in ("hadm_id", "admittime", "los_days")]
print(f"Target  : los_days  mean={df['los_days'].mean():.2f}  median={df['los_days'].median():.2f}")
print(f"Features: {len(feature_cols)} total")
print(f"Missing : {df[feature_cols].isna().sum().sum()}")
