"""
Synthea LOS Dataset Builder (via Neo4j)
========================================
Produces processed/synthea_los.csv — one row per Inpatient encounter:
  - los_days          : target (Length of Stay in days)
  - demographics      : age, gender, marital, income
  - time features     : admission day/month/year, is_weekend
  - 31 Elixhauser-style comorbidity flags (cm_*) via SNOMED description matching
  - count features    : num_diagnoses, num_procedures, num_drugs
  - encounter cost    : total_cost

Run:
    source ~/my_python_env/venv/bin/activate
    python build_synthea_los_neo4j.py
"""

import os
import re
import numpy as np
import pandas as pd
from neo4j import GraphDatabase

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = "synthea-sample"
OUT            = "processed"
os.makedirs(OUT, exist_ok=True)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def query(cypher: str) -> pd.DataFrame:
    with driver.session(database=NEO4J_DATABASE) as s:
        result = s.run(cypher)
        return pd.DataFrame([dict(r) for r in result])


# ============================================================
# Elixhauser comorbidity patterns via SNOMED description text
# (case-insensitive substring/regex matching on description field)
# ============================================================
elixhauser_text = {
    "chf":              ["heart failure", "cardiac failure", "congestive heart"],
    "arrhythmia":       ["arrhythmia", "atrial fibrillation", "atrial flutter",
                         "ventricular tachycardia", "heart block", "sick sinus"],
    "valve":            ["valvular", "mitral stenosis", "mitral regurgitation",
                         "aortic stenosis", "aortic regurgitation", "tricuspid",
                         "endocarditis"],
    "pulm_circ":        ["pulmonary embolism", "pulmonary hypertension",
                         "cor pulmonale"],
    "pvd":              ["peripheral vascular", "peripheral arterial",
                         "atherosclerosis", "intermittent claudication",
                         "aortic aneurysm"],
    "htn_uncomp":       ["essential hypertension", "primary hypertension",
                         "hypertension (disorder)"],
    "htn_comp":         ["hypertensive heart", "hypertensive renal",
                         "hypertensive chronic kidney", "hypertensive disease"],
    "paralysis":        ["paralysis", "hemiplegia", "paraplegia",
                         "quadriplegia", "hemiparesis"],
    "neuro_other":      ["parkinson", "multiple sclerosis", "epilepsy",
                         "dementia", "alzheimer", "cerebral palsy",
                         "huntington", "motor neuron"],
    "copd":             ["chronic obstructive pulmonary", "emphysema",
                         "chronic bronchitis", "bronchiectasis"],
    "dm_uncomp":        ["diabetes mellitus type 2", "type 2 diabetes",
                         "type ii diabetes", "non-insulin-dependent diabetes"],
    "dm_comp":          ["diabetic nephropathy", "diabetic retinopathy",
                         "diabetic neuropathy", "diabetic ketoacidosis",
                         "diabetic foot", "diabetic macular"],
    "hypothyroid":      ["hypothyroidism", "myxedema", "hashimoto"],
    "renal":            ["renal failure", "chronic kidney disease",
                         "end stage renal", "kidney failure", "renal transplant"],
    "liver":            ["liver failure", "cirrhosis", "hepatic failure",
                         "portal hypertension", "hepatitis b", "hepatitis c",
                         "alcoholic liver"],
    "ulcer":            ["peptic ulcer", "gastric ulcer", "duodenal ulcer",
                         "gastrointestinal ulcer"],
    "hiv":              ["human immunodeficiency virus", "hiv infection",
                         "acquired immunodeficiency"],
    "lymphoma":         ["lymphoma", "hodgkin", "non-hodgkin"],
    "metastatic":       ["metastatic", "metastasis", "secondary malignant"],
    "solid_tumor":      ["carcinoma", "adenocarcinoma", "malignant neoplasm",
                         "malignant tumor", "melanoma", "sarcoma"],
    "rheumatoid":       ["rheumatoid arthritis", "systemic lupus",
                         "scleroderma", "vasculitis", "polymyositis",
                         "ankylosing spondylitis"],
    "coagulopathy":     ["coagulopathy", "thrombocytopenia", "hemophilia",
                         "disseminated intravascular coagulation",
                         "von willebrand"],
    "obesity":          ["obesity", "morbid obesity"],
    "weight_loss":      ["malnutrition", "cachexia", "protein-energy",
                         "failure to thrive"],
    "fluid_electrolyte":["hyponatremia", "hypernatremia", "hypokalemia",
                         "hyperkalemia", "electrolyte imbalance",
                         "fluid overload", "dehydration"],
    "blood_loss":       ["blood loss anemia", "acute blood loss",
                         "hemorrhagic anemia"],
    "anemia":           ["iron deficiency anemia", "vitamin b12 deficiency anemia",
                         "folate deficiency anemia", "aplastic anemia",
                         "hemolytic anemia", "sickle cell", "thalassemia"],
    "alcohol":          ["alcohol use disorder", "alcoholism",
                         "alcohol dependence", "alcohol abuse"],
    "drug":             ["opioid use disorder", "cocaine dependence",
                         "substance use disorder", "drug abuse",
                         "cannabis dependence", "heroin"],
    "psychosis":        ["schizophrenia", "schizoaffective", "psychosis",
                         "bipolar i disorder", "delusional disorder"],
    "depression":       ["major depression", "depressive disorder",
                         "persistent depressive", "dysthymia"],
}

def build_elixhauser_regex():
    """Compile one regex per comorbidity from text pattern lists."""
    compiled = {}
    for cm, terms in elixhauser_text.items():
        pattern = "|".join(re.escape(t) for t in terms)
        compiled[cm] = re.compile(pattern, re.IGNORECASE)
    return compiled

cm_regex = build_elixhauser_regex()


def match_elixhauser(descriptions: list[str]) -> dict:
    """Given a list of SNOMED descriptions for one encounter, return cm_* flags."""
    combined = " | ".join(str(d).lower() for d in descriptions)
    return {
        f"cm_{cm}": int(bool(rx.search(combined)))
        for cm, rx in cm_regex.items()
    }


# ============================================================
# 1. Inpatient encounters + patient demographics
# ============================================================
print("1. Inpatient encounters + patients ...")
enc_df = query("""
MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e:Inpatient)
RETURN
  e.id            AS encounter_id,
  p.id            AS patient_id,
  p.age           AS age,
  p.gender        AS gender,
  p.marital       AS marital,
  p.income        AS income,
  e.date          AS admittime,
  e.end           AS dischtime,
  e.totalCost     AS total_cost,
  e.description   AS encounter_desc
""")

enc_df["admittime"] = pd.to_datetime(enc_df["admittime"].astype(str))
enc_df["dischtime"] = pd.to_datetime(enc_df["dischtime"].astype(str))
enc_df["los_days"]  = (enc_df["dischtime"] - enc_df["admittime"]).dt.total_seconds() / 86400.0

enc_df = enc_df[(enc_df["los_days"] > 0) & (enc_df["los_days"] <= 365)].copy()
print(f"   {len(enc_df):,} inpatient encounters after filter")

# Time features
enc_df["admission_dayofweek"] = enc_df["admittime"].dt.weekday
enc_df["admission_month"]     = enc_df["admittime"].dt.month
enc_df["admission_year"]      = enc_df["admittime"].dt.year
enc_df["is_weekend"]          = enc_df["admittime"].dt.weekday.isin([5, 6]).astype(np.int8)

# Demographics
enc_df["gender_bin"]   = enc_df["gender"].str.upper().map({"F": 0, "M": 1}).fillna(0).astype(np.int8)
enc_df["is_married"]   = enc_df["marital"].str.upper().eq("M").astype(np.int8)
enc_df["anchor_age"]   = pd.to_numeric(enc_df["age"], errors="coerce").fillna(0)
enc_df["income"]       = pd.to_numeric(enc_df["income"], errors="coerce").fillna(0)
enc_df["total_cost"]   = pd.to_numeric(enc_df["total_cost"], errors="coerce").fillna(0)


# ============================================================
# 2. Diagnoses per encounter → Elixhauser flags + count
# ============================================================
print("2. Diagnoses → Elixhauser comorbidity flags ...")
diag_df = query("""
MATCH (e:Inpatient)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
RETURN e.id AS encounter_id, d.code AS snomed_code, d.description AS description
""")

num_diagnoses = (
    diag_df.groupby("encounter_id")
    .size()
    .rename("num_diagnoses")
    .reset_index()
)

# Group descriptions per encounter, apply regex matching
diag_grouped = diag_df.groupby("encounter_id")["description"].apply(list)

print("   Applying Elixhauser text patterns ...")
cm_records = []
for enc_id, descs in diag_grouped.items():
    row = {"encounter_id": enc_id}
    row.update(match_elixhauser(descs))
    cm_records.append(row)

cm_df = pd.DataFrame(cm_records)
cm_cols = [c for c in cm_df.columns if c.startswith("cm_")]
cm_df[cm_cols] = cm_df[cm_cols].astype(np.int8)


# ============================================================
# 3. Procedure count
# ============================================================
print("3. Procedure count ...")
proc_df = query("""
MATCH (e:Inpatient)-[:HAS_PROCEDURE]->(pr:Procedure)
RETURN e.id AS encounter_id, count(pr) AS num_procedures
""")


# ============================================================
# 4. Drug count
# ============================================================
print("4. Drug count ...")
drug_df = query("""
MATCH (e:Inpatient)-[:HAS_DRUG]->(dr:Drug)
RETURN e.id AS encounter_id, count(dr) AS num_drugs
""")


# ============================================================
# 5. Merge
# ============================================================
print("5. Merging ...")
df = enc_df[[
    "encounter_id", "patient_id", "admittime", "los_days",
    "anchor_age", "gender_bin", "is_married", "income",
    "admission_dayofweek", "admission_month", "admission_year", "is_weekend",
    "total_cost",
]].copy()

for tbl in [num_diagnoses, cm_df, proc_df, drug_df]:
    df = df.merge(tbl, on="encounter_id", how="left")

count_cols = ["num_diagnoses", "num_procedures", "num_drugs"]
df[count_cols] = df[count_cols].fillna(0).astype(int)
df[cm_cols]    = df[cm_cols].fillna(0).astype(np.int8)

df = df.sort_values("admittime").reset_index(drop=True)

# ============================================================
# 6. Save
# ============================================================
out_path = f"{OUT}/synthea_los.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

feature_cols = [c for c in df.columns
                if c not in ("encounter_id", "patient_id", "admittime", "los_days")]
print(f"Target  : los_days  mean={df['los_days'].mean():.2f}  median={df['los_days'].median():.2f}")
print(f"Features: {len(feature_cols)} total")
print(f"Missing : {df[feature_cols].isna().sum().sum()}")

print("\nComorbidity prevalence:")
for c in cm_cols:
    if df[c].sum() > 0:
        print(f"  {c:<28} {df[c].mean():.1%}")

driver.close()
