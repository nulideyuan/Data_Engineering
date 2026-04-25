"""
MIMIC-4 LOS Dataset Builder
============================
Produces processed/mimic4_los.csv — one row per inpatient admission, with:
  - los_days             : target (Length of Stay in days)
  - demographics         : age, gender, insurance, admission context
  - 31 Elixhauser flags  : cm_<name> binary comorbidity indicators
  - count features       : num_diagnoses, num_procedures, num_transfers,
                           num_prescriptions, num_labs
  - ICU flag             : has_icu
  - ED triage vitals     : triage_* (NaN→0 for non-ED patients), had_ed_visit

Designed for LOS regression + SHAP interpretability.
labevents (~17 GB) and prescriptions (~3.4 GB) are read in chunks.
"""

import os
import re
import numpy as np
import pandas as pd

RAW = "raw/mimic4"
OUT = "processed"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Elixhauser comorbidity patterns (ICD-9 + ICD-10 prefixes)
# ============================================================
elixhauser_patterns = {
    "chf": [
        "39891","40201","40211","40291",
        "40401","40403","40411","40413","40491","40493",
        "4254","4255","4256","4257","4258","4259",
        "428","I50","I43"
    ],
    "arrhythmia": [
        "426","427","7850","99601","99604",
        "V450","V533",
        "I44","I45","I47","I48","I49",
        "R000","R001","R008",
        "T821","Z450","Z950"
    ],
    "valve": [
        "0932","394","395","396","397","424",
        "7463","7464","7465","7466",
        "V422","V433",
        "A520",
        "I05","I06","I07","I08",
        "I091","I098",
        "I34","I35","I36","I37","I38","I39",
        "Q230","Q231","Q232","Q233",
        "Z952","Z954"
    ],
    "pulm_circ": [
        "4150","4151","416","417",
        "I26","I27","I280","I288","I289"
    ],
    "pvd": [
        "0930","4373","440","441","442",
        "4431","4432","4433","4434","4435","4436","4437","4438","4439",
        "4471","5571","5579","V434",
        "I70","I71","I731","I738","I739",
        "I771","I790","I792",
        "K551","K558","K559",
        "Z958","Z959"
    ],
    "htn_uncomp": ["4011","4019","6420","I10"],
    "htn_comp": [
        "4010","402","403","404","405",
        "6421","6422","6427","6429",
        "I11","I12","I13","I15"
    ],
    "paralysis": [
        "3341","342","343","344",
        "4382","4383","4384","4385",
        "G041","G114","G801","G802",
        "G81","G82",
        "G830","G831","G832","G833","G834","G839"
    ],
    "neuro_other": [
        "330","331","332","3334","3335","33392",
        "334","335","3362","340","341",
        "345","347","3481","3483",
        "7803","7843",
        "G10","G11","G12","G13",
        "G20","G21","G22",
        "G254","G255",
        "G312","G318","G319",
        "G32",
        "G35","G36","G37",
        "G40","G41",
        "G931","G934",
        "R470","R56"
    ],
    "copd": [
        "490","491","492","493","494","495","496","497","498","499",
        "500","501","502","503","504","505",
        "5064","5081","5088",
        "I278","I279",
        "J40","J41","J42","J43","J44","J45","J46","J47",
        "J60","J61","J62","J63","J64","J65","J66","J67",
        "J684","J701","J703"
    ],
    "dm_uncomp": ["2500","2501","2502","2503","6480"],
    "dm_comp": [
        "2504","2505","2506","2507","2508","2509","7751",
        "E102","E103","E104","E105","E106","E107","E108",
        "E112","E113","E114","E115","E116","E117","E118",
        "E122","E123","E124","E125","E126","E127","E128",
        "E132","E133","E134","E135","E136","E137","E138",
        "E142","E143","E144","E145","E146","E147","E148"
    ],
    "hypothyroid": [
        "2409","243","244","2461","2468",
        "E00","E01","E02","E03","E890"
    ],
    "renal": [
        "40301","40311","40391",
        "40402","40403","40412","40413","40492","40493",
        "585","586","5880",
        "V420","V451","V56",
        "I120","I131",
        "N18","N19","N250",
        "Z490","Z491","Z492",
        "Z940","Z992"
    ],
    "liver": [
        "07022","07023","07032","07033","07044","07054","0706","0709",
        "4560","4561","4562",
        "570","571",
        "5722","5723","5724","5725","5726","5727","5728",
        "5733","5734","5738","5739",
        "V427",
        "B18",
        "I85","I864","I982",
        "K70","K711","K713","K714","K715","K717",
        "K72","K73","K74",
        "K760","K762","K763","K764","K765","K766","K767","K768","K769",
        "Z944"
    ],
    "ulcer": [
        "5317","5319","5327","5329","5337","5339","5347","5349",
        "V1271",
        "K257","K259","K267","K269","K277","K279","K287","K289"
    ],
    "hiv": ["042","043","044","B20","B21","B22","B24"],
    "lymphoma": [
        "200","201","202","2030","2386","2733",
        "V1071","V1072","V1079",
        "C81","C82","C83","C84","C85","C88","C96","C900","C902"
    ],
    "metastatic": ["196","197","198","199","C77","C78","C79","C80"],
    "solid_tumor": [
        "140","141","142","143","144","145","146","147","148","149",
        "150","151","152","153","154","155","156","157","158","159",
        "160","161","162","163","164","165","166","167","168","169",
        "170","171","172",
        "174","175","179","180","181","182","183","184","185","186","187","188","189",
        "190","191","192","193","194","195",
        "V10",
        "C00","C01","C02","C03","C04","C05","C06","C07","C08","C09",
        "C10","C11","C12","C13","C14","C15","C16","C17","C18","C19",
        "C20","C21","C22","C23","C24","C25","C26",
        "C30","C31","C32","C33","C34",
        "C37","C38","C39","C40","C41",
        "C43",
        "C45","C46","C47","C48","C49","C50","C51","C52","C53","C54",
        "C55","C56","C57","C58",
        "C60","C61","C62","C63","C64","C65","C66","C67","C68","C69",
        "C70","C71","C72","C73","C74","C75","C76",
        "C97"
    ],
    "rheumatoid": [
        "446","7010","710","7112","714","7193","720","725",
        "7285","72889","72930",
        "L940","L941","L943",
        "M05","M06","M08",
        "M120","M123",
        "M30",
        "M310","M311","M312","M313",
        "M32","M33","M34","M35",
        "M45","M461","M468","M469"
    ],
    "coagulopathy": [
        "286","2871","2873","2874","2875",
        "D65","D66","D67","D68",
        "D691","D693","D694","D695","D696"
    ],
    "obesity": ["2780","E66"],
    "weight_loss": [
        "260","261","262","263","7832","7994",
        "E40","E41","E42","E43","E44","E45","E46",
        "R634","R64"
    ],
    "fluid_electrolyte": ["2536","276","E222","E86","E87"],
    "blood_loss": ["2800","6482","D500"],
    "anemia": [
        "2801","2802","2803","2804","2805","2806","2807","2808","2809",
        "281",
        "2852","2859",
        "D508","D509","D51","D52","D53"
    ],
    "alcohol": [
        "2652","291","303","3050",
        "3575","4255","5353",
        "5710","5711","5712","5713",
        "980","V113",
        "F10","E52","G621","I426",
        "K292","K700","K703","K709",
        "T51","Z502","Z714","Z721"
    ],
    "drug": [
        "292","304",
        "3052","3053","3054","3055","3056","3057","3058","3059",
        "V6542",
        "F11","F12","F13","F14","F15","F16","F18","F19",
        "Z715","Z722"
    ],
    "psychosis": [
        "2938","295",
        "29604","29614","29644","29654",
        "297","298",
        "F20","F22","F23","F24","F25","F28","F29",
        "F302","F312","F315"
    ],
    "depression": [
        "2962","2963","2965",
        "3004","30112","309","311",
        "F204",
        "F313","F314","F315",
        "F32","F33",
        "F341","F412","F432"
    ],
}


# Pre-compile one prefix regex per comorbidity (all codes, no version split)
_cm_regex = {}
for _cm, _codes in elixhauser_patterns.items():
    _sorted = sorted(set(re.escape(c) for c in _codes), key=len, reverse=True)
    _cm_regex[_cm] = re.compile("^(?:" + "|".join(_sorted) + ")")


def compute_elixhauser(diag: pd.DataFrame) -> pd.DataFrame:
    """
    Given diagnoses_icd (columns: hadm_id, icd_code str),
    return DataFrame with hadm_id + 31 cm_* binary columns (int8).
    Matches all ICD codes (9 and 10) via prefix against elixhauser_patterns.
    """
    all_hadm = diag["hadm_id"].unique()
    cm_df = pd.DataFrame({"hadm_id": all_hadm})
    for cm, rx in _cm_regex.items():
        mask = diag["icd_code"].str.match(rx, na=False)
        matched = set(diag.loc[mask, "hadm_id"])
        cm_df[f"cm_{cm}"] = cm_df["hadm_id"].isin(matched).astype(np.int8)
    return cm_df


# ============================================================
# 1. Admissions
# ============================================================
print("Loading admissions ...")
adm = pd.read_csv(
    f"{RAW}/admissions.csv",
    parse_dates=["admittime", "dischtime"],
    usecols=[
        "subject_id", "hadm_id", "admittime", "dischtime",
        "admission_type", "admission_location", "insurance",
        "hospital_expire_flag",
    ],
)

adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400.0

# Keep only valid inpatient stays
adm = adm[(adm["los_days"] > 0) & (adm["los_days"] <= 365)].copy()
print(f"  After LOS filter: {len(adm):,} admissions")

# Temporal features (using admittime)
adm["admission_dayofweek"] = adm["admittime"].dt.weekday   # 0=Mon
adm["admission_month"] = adm["admittime"].dt.month
adm["admission_year"] = adm["admittime"].dt.year
adm["is_weekend"] = adm["admittime"].dt.weekday.isin([5, 6]).astype(np.int8)

# Emergency admission
emergency_types = {"EW EMER.", "DIRECT EMER.", "URGENT"}
adm["is_emergency"] = adm["admission_type"].isin(emergency_types).astype(np.int8)

# Insurance (ordinal: Medicare=0, Medicaid=1, Private=2, Other=3)
ins_map = {"Medicare": 0, "Medicaid": 1, "Private": 2, "Other": 3, "No charge": 3}
adm["insurance_cat"] = adm["insurance"].map(ins_map).fillna(3).astype(np.int8)

# Admission location (label encoded, sorted by frequency for interpretability)
loc_order = (
    adm["admission_location"]
    .value_counts()
    .index.tolist()
)
loc_map = {loc: i for i, loc in enumerate(loc_order)}
adm["admission_location_cat"] = adm["admission_location"].map(loc_map).fillna(len(loc_order)).astype(np.int8)

# ============================================================
# 2. Patients (age, gender)
# ============================================================
print("Loading patients ...")
pat = pd.read_csv(
    f"{RAW}/patients.csv",
    usecols=["subject_id", "gender", "anchor_age"],
)
pat["gender_bin"] = pat["gender"].str.upper().map({"F": 0, "M": 1}).fillna(0).astype(np.int8)

# ============================================================
# 3. Diagnoses → Elixhauser comorbidities + count
# ============================================================
print("Loading diagnoses_icd ...")
diag = pd.read_csv(
    f"{RAW}/diagnoses_icd.csv",
    dtype={"icd_code": str},
    usecols=["hadm_id", "icd_code", "icd_version"],
)

num_diagnoses = diag.groupby("hadm_id").size().rename("num_diagnoses").reset_index()

print("Computing Elixhauser comorbidities ...")
cm_df = compute_elixhauser(diag)
del diag

# ============================================================
# 4. Procedures (count)
# ============================================================
print("Loading procedures_icd ...")
proc = pd.read_csv(f"{RAW}/procedures_icd.csv", usecols=["hadm_id"])
num_procedures = proc.groupby("hadm_id").size().rename("num_procedures").reset_index()
del proc

# ============================================================
# 5. Transfers (count + ICU flag)
# ============================================================
print("Loading transfers ...")
trans = pd.read_csv(f"{RAW}/transfers.csv", usecols=["hadm_id", "careunit"])

num_transfers = trans.groupby("hadm_id").size().rename("num_transfers").reset_index()

icu_mask = trans["careunit"].str.contains("Intensive|ICU|CCU", case=False, na=False)
has_icu = (
    trans[icu_mask]
    .groupby("hadm_id")
    .size()
    .rename("has_icu")
    .gt(0)
    .astype(np.int8)
    .reset_index()
)
del trans

# ============================================================
# 6. ED stays + Triage vitals
# ============================================================
print("Loading edstays + triage ...")
ed = pd.read_csv(f"{RAW}/edstays.csv", usecols=["hadm_id", "stay_id"])
triage = pd.read_csv(
    f"{RAW}/triage.csv",
    usecols=["stay_id", "temperature", "heartrate", "resprate",
             "o2sat", "sbp", "dbp", "pain", "acuity"],
)

ed_triage = ed.merge(triage, on="stay_id", how="inner")

# Force all vital columns to numeric (pain is stored as string in MIMIC-4)
for col in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]:
    ed_triage[col] = pd.to_numeric(ed_triage[col], errors="coerce")

ed_vitals = (
    ed_triage
    .groupby("hadm_id")
    .agg(
        triage_temperature=("temperature", "mean"),
        triage_heartrate=("heartrate", "mean"),
        triage_resprate=("resprate", "mean"),
        triage_o2sat=("o2sat", "mean"),
        triage_sbp=("sbp", "mean"),
        triage_dbp=("dbp", "mean"),
        triage_pain=("pain", "mean"),
        triage_acuity=("acuity", "mean"),
    )
    .reset_index()
)
ed_vitals["had_ed_visit"] = np.int8(1)
del ed, triage, ed_triage

# ============================================================
# 7. Prescriptions (count — chunked, ~3.4 GB)
# ============================================================
print("Loading prescriptions (chunked) ...")
rx_acc = {}
for chunk in pd.read_csv(f"{RAW}/prescriptions.csv", usecols=["hadm_id"], chunksize=500_000):
    for hid, cnt in chunk["hadm_id"].value_counts().items():
        rx_acc[hid] = rx_acc.get(hid, 0) + cnt

num_prescriptions = (
    pd.Series(rx_acc, name="num_prescriptions")
    .rename_axis("hadm_id")
    .reset_index()
)
del rx_acc

# ============================================================
# 8. Lab events (count — chunked, ~17 GB)
# ============================================================
print("Loading labevents (chunked) — this may take several minutes ...")
lab_acc = {}
for chunk in pd.read_csv(
    f"{RAW}/labevents.csv",
    usecols=["hadm_id"],
    chunksize=500_000,
    low_memory=False,
):
    chunk = chunk.dropna(subset=["hadm_id"])
    chunk["hadm_id"] = chunk["hadm_id"].astype(int)
    for hid, cnt in chunk["hadm_id"].value_counts().items():
        lab_acc[hid] = lab_acc.get(hid, 0) + cnt

num_labs = (
    pd.Series(lab_acc, name="num_labs")
    .rename_axis("hadm_id")
    .reset_index()
)
del lab_acc

# ============================================================
# 9. Merge everything → final flat table
# ============================================================
print("Merging all features ...")

df = adm.merge(pat[["subject_id", "gender_bin", "anchor_age"]], on="subject_id", how="left")
df = df.merge(num_diagnoses, on="hadm_id", how="left")
df = df.merge(cm_df,          on="hadm_id", how="left")
df = df.merge(num_procedures, on="hadm_id", how="left")
df = df.merge(num_transfers,  on="hadm_id", how="left")
df = df.merge(has_icu,        on="hadm_id", how="left")
df = df.merge(ed_vitals,      on="hadm_id", how="left")
df = df.merge(num_prescriptions, on="hadm_id", how="left")
df = df.merge(num_labs,       on="hadm_id", how="left")

# Fill missing counts with 0
count_cols = [
    "num_diagnoses", "num_procedures", "num_transfers",
    "num_prescriptions", "num_labs",
]
df[count_cols] = df[count_cols].fillna(0).astype(int)

# Fill missing ICU flag and ED flag with 0
df["has_icu"] = df["has_icu"].fillna(0).astype(np.int8)
df["had_ed_visit"] = df["had_ed_visit"].fillna(0).astype(np.int8)

# ED vitals: leave NaN for non-ED patients (models can impute or use 0)
# But also add a simple fill=0 version for tree models
triage_cols = [
    "triage_temperature", "triage_heartrate", "triage_resprate",
    "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
]
df[triage_cols] = df[triage_cols].fillna(0)

# Fill Elixhauser NaN with 0 (admissions not in diagnoses table)
cm_cols = [c for c in df.columns if c.startswith("cm_")]
df[cm_cols] = df[cm_cols].fillna(0).astype(np.int8)

# Drop raw string columns not needed downstream
df = df.drop(columns=["admission_type", "admission_location", "insurance",
                       "dischtime", "subject_id"],
             errors="ignore")

# Sort by admission time (important for temporal train/val/test splits)
df = df.sort_values("admittime").reset_index(drop=True)

# ============================================================
# 10. Save
# ============================================================
out_path = f"{OUT}/mimic4_los.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================
# 11. Feature summary
# ============================================================
feature_cols = [
    c for c in df.columns
    if c not in ("hadm_id", "admittime", "los_days")
]
continuous_cols = [
    "anchor_age", "admission_dayofweek", "admission_month", "admission_year",
    "num_diagnoses", "num_procedures", "num_transfers",
    "num_prescriptions", "num_labs",
] + triage_cols

binary_cols = [c for c in feature_cols if c not in continuous_cols]

print(f"\nTarget  : los_days  (mean={df['los_days'].mean():.2f}, median={df['los_days'].median():.2f})")
print(f"Features: {len(feature_cols)} total")
print(f"  Continuous : {len(continuous_cols)}")
print(f"  Binary     : {len(binary_cols)}")
print(f"\nElixhauser comorbidity columns ({len(cm_cols)}):")
print("  " + ", ".join(cm_cols))
print(f"\nMissing values per column:\n{df[feature_cols].isna().sum()[df[feature_cols].isna().sum() > 0]}")
