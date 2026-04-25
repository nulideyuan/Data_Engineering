# LOS Prediction Datasets

Four processed datasets for Length of Stay prediction and SHAP interpretability analysis.

| Dataset | File | Rows | Columns | Source | Population |
|---|---|---|---|---|---|
| MIMIC-4 | `mimic4_los.csv` | 545,847 | 59 | Local CSV files | General inpatient |
| MIMIC-3 | `mimic3_los.csv` | 58,878 | 50 | Google BigQuery | ICU patients |
| eICU | `eicu_los.csv` | 94,861 | 40 | Google BigQuery | ICU patients (multi-site) |
| Synthea | `synthea_los.csv` | 6,480 | 47 | Neo4j (local) | Synthetic inpatient |

---

## How to Build

### MIMIC-4
```bash
python build_mimic4_los_dataset.py
```
Reads directly from `raw/mimic4/*.csv`. `labevents.csv` (~17 GB) is processed in chunks automatically.

### MIMIC-3
```bash
source ~/my_python_env/venv/bin/activate
python build_mimic3_los_bigquery.py
```
Queries `physionet-data.mimiciii_clinical` via Google BigQuery (project `eicu-494316`). No local file download needed.

Requirements:
- PhysioNet credentialed access to MIMIC-III
- BigQuery access granted via PhysioNet MIMIC-III page
- `gcloud auth application-default login` completed

### eICU
```bash
source ~/my_python_env/venv/bin/activate
python build_eicu_los_bigquery.py
```
Queries `physionet-data.eicu_crd` via Google BigQuery (project `eicu-494316`).

Requirements:
- PhysioNet credentialed access to eICU-CRD
- BigQuery access granted via PhysioNet eICU page
- `gcloud auth application-default login` completed

### Synthea (Neo4j)
```bash
source ~/my_python_env/venv/bin/activate
python build_synthea_los_neo4j.py
```
Queries the local Neo4j database `synthea-sample` at `neo4j://127.0.0.1:7687`.

Requirements:
- Neo4j running locally with `synthea-sample` database loaded
- `neo4j` Python package installed (`pip install neo4j`)

---

## How to Load

All datasets use the same load function in `dataset.py`:

```python
from dataset import load_mimic4_los

(X_train, y_train,
 X_val,   y_val,
 X_test,  y_test,
 feature_cols,     # same order as X columns — pass directly to SHAP
 continuous_cols,  # RobustScaled
 binary_cols,      # not scaled
 scaler) = load_mimic4_los(csv_path="processed/mimic4_los.csv")

# MIMIC-3: csv_path="processed/mimic3_los.csv"
# Synthea:  csv_path="processed/synthea_los.csv", target="los_days"
```

Split: time-based 70 / 10 / 20 sorted by `admittime`.

---

## Dataset Comparison

| Property | MIMIC-4 | MIMIC-3 | eICU | Synthea |
|---|---|---|---|---|
| LOS mean | 4.76 days | 10.15 days | 3.15 days | 4.92 days |
| LOS median | 2.82 days | 6.49 days | 1.89 days | 4.04 days |
| LOS unit | Hospital stay | Hospital stay | ICU stay | Hospital stay |
| ICU rate | 15.6% | 98% | 100% | — |
| ED triage vitals | Yes | No | No | No |
| Diagnosis coding | ICD-9 + ICD-10 | ICD-9 only | None (APACHE) | SNOMED CT |
| Comorbidity method | Elixhauser (ICD prefix) | Elixhauser (ICD prefix) | APACHE `apachepredvar` fields | Elixhauser (SNOMED text) |
| Severity score | — | — | APACHE score + pred mortality | — |
| Physiology at admission | Triage vitals (ED only) | — | 15 APACHE APS variables | — |
| Data type | Real patients | Real ICU patients | Real ICU patients (multi-site) | Synthetic patients |

---

## Comorbidity Logic

The `cm_*` columns represent the same 31 Elixhauser comorbidity categories across all datasets, but are derived differently depending on what each database provides.

### What is Elixhauser?

The Elixhauser comorbidity index is a standard set of 31 clinical categories (CHF, diabetes, renal failure, etc.) used widely in hospital outcomes research. For each admission, each category becomes a binary flag: does this patient have this condition, based on diagnosis codes recorded at admission.

### Pattern codes and prefix matching

Each comorbidity has a list of ICD pattern codes. Some are **exact** codes; others are **category prefixes** that represent a whole subtree of the ICD hierarchy:

```
"chf": ["39891", "40201", "428", "I50", "I43", ...]
```

- `"39891"` — exact ICD-9 code, matches only `39891`
- `"428"` — prefix, matches `4280`, `4281`, `42821`, `42823`, ... (all heart failure subcategories)
- `"I50"` — prefix, matches `I500`, `I501`, `I509`, ... (ICD-10 heart failure)

This works because ICD is a hierarchical tree. A parent code like `428` always prefixes all its children.

The matching regex for each comorbidity is compiled once at build time, patterns sorted longest-first to prevent shorter prefixes from short-circuiting longer exact codes:

```
^(?:40493|40491|...|39891|I50|I43|428)
```

Any admission with at least one matching diagnosis code gets `cm_chf = 1`.

### MIMIC-4 — ICD prefix matching (ICD-9 + ICD-10)

MIMIC-4 stores both ICD-9 and ICD-10 codes in `diagnoses_icd` (mixed per admission). A single unified regex per comorbidity is applied to all codes regardless of version — the pattern codes themselves are unambiguous (`428` only exists in ICD-9 space; `I50` only in ICD-10 space, so there is no cross-version false-positive risk).

```
"4280"  → starts with "428"  → cm_chf = 1
"I5009" → starts with "I50"  → cm_chf = 1
```

### MIMIC-3 — ICD prefix matching (ICD-9 only)

Same prefix-matching approach. MIMIC-3 contains only ICD-9 codes, so the pattern list (`elixhauser_icd9`) contains only ICD-9 entries — no ICD-10 codes included.

```
"42830" → starts with "428" → cm_chf = 1
```

### eICU — APACHE predictor variables (direct comorbidity fields)

eICU does not use ICD codes. Instead, the APACHE scoring system records comorbidities directly in `apachepredvar` at ICU admission. Only 10 comorbidities are available (fewer than the 31 Elixhauser categories):

```
apachepredvar.diabetes = 1
  → cm_diabetes = 1
```

These are more reliable than ICD-derived flags because they are actively collected during APACHE scoring, not inferred from billing codes. The column names use `cm_` prefix for consistency but map to APACHE-defined categories, not Elixhauser.

eICU also provides continuous physiological variables from `apacheapsvar` (prefixed `apache_`) and an overall `apache_score`. In the raw eICU data, `-1` is a sentinel value meaning "not measured" (not a real physiological reading). All `-1` values are converted to `NaN` before saving. Rows with any missing APACHE value are dropped (104,766 rows removed), leaving only patients with complete APACHE profiles. Seven blood-gas/liver variables with >50% missingness are dropped entirely as they are only collected for intubated or critically ill subsets:

Dropped: `apache_ph`, `apache_pao2`, `apache_pco2`, `apache_fio2`, `apache_bilirubin`, `apache_albumin`, `apache_urine`

### Synthea (Neo4j) — SNOMED description text matching

Synthea stores diagnoses as SNOMED CT codes with text descriptions (e.g., `"Chronic congestive heart failure (disorder)"`). Since there are no ICD codes, comorbidities are detected by **case-insensitive regex substring matching against the description field**:

```
Diagnosis.description = "Chronic congestive heart failure (disorder)"
  → matches "congestive heart" in chf_patterns
  → cm_chf = 1
```

For each inpatient encounter, all diagnoses from the patient's **full history** (not just that encounter) are collected and matched — this is necessary because Synthea records diagnoses at first occurrence only, not repeated on every encounter.

**Coverage audit:** The Synthea sample database contains exactly 45 unique SNOMED descriptions. All descriptions that correspond to Elixhauser categories are matched correctly. The descriptions that produce no match (e.g., Hyperlipidemia, Osteoporosis, Sepsis, Viral sinusitis, Atopic dermatitis) are genuinely outside the 31 Elixhauser categories — they are acute or non-Elixhauser conditions, so zero flags for those encounters is correct.

| Synthea description | Matched category |
|---|---|
| Chronic congestive heart failure (disorder) | `cm_chf` |
| Atrial fibrillation (disorder) | `cm_arrhythmia` |
| Acute pulmonary embolism (disorder) | `cm_pulm_circ` |
| Hypertension | `cm_htn_uncomp` |
| Alzheimer's disease (disorder) | `cm_neuro_other` |
| Asthma / Childhood asthma / Pulmonary emphysema | `cm_copd` |
| Diabetes / Prediabetes | `cm_dm_uncomp` |
| History of renal transplant (situation) | `cm_renal` |
| Human immunodeficiency virus infection (disorder) | `cm_hiv` |
| Acute myeloid leukemia disease (disorder) | `cm_lymphoma` |
| Malignant neoplasm / Malignant tumor / Carcinoma / Neoplasm of | `cm_solid_tumor` |
| Lupus erythematosus / Rheumatoid arthritis | `cm_rheumatoid` |
| Anemia (disorder) | `cm_anemia` |
| Opioid abuse (disorder) | `cm_drug` |
| Major depression single episode | `cm_depression` |

The `cm_lymphoma` pattern includes "leukemia" and "myeloma" in addition to "lymphoma" and "hodgkin", matching the original Elixhauser definition which covers all hematologic malignancies.

**Why not SNOMED CT hierarchy matching?** A more precise approach would use the SNOMED IS-A concept hierarchy (e.g., find all descendants of concept 84114007 "Heart failure") rather than text patterns. This requires loading the full SNOMED CT RF2 release (~2 GB) into a local database. For this Synthea sample with 45 known descriptions, text matching achieves equivalent coverage and the added complexity is not justified.

---

## Column Reference

### IDs / Time (not features)

| Column | MIMIC-4 | MIMIC-3 | eICU | Synthea | Description |
|---|---|---|---|---|---|
| `hadm_id` | ✅ | ✅ | — | — | Hospital admission ID |
| `patientunitstayid` | — | — | ✅ | — | eICU ICU stay ID |
| `encounter_id` | — | — | — | ✅ | Synthea encounter UUID |
| `patient_id` | — | — | — | ✅ | Synthea patient UUID |
| `admittime` | ✅ | ✅ | — | ✅ | Admission datetime — used for time-based split |

### Target

| Column | MIMIC-4 | MIMIC-3 | eICU | Synthea | Description |
|---|---|---|---|---|---|
| `los_days` | ✅ | ✅ | — | ✅ | Hospital length of stay in days |
| `icu_los_days` | — | — | ✅ | — | ICU length of stay in days (filtered 0–90, complete APACHE only) |

### Demographics

| Column | MIMIC-4 | MIMIC-3 | eICU | Synthea | Description |
|---|---|---|---|---|---|
| `anchor_age` | ✅ | ✅ | ✅ | ✅ | Age at admission (MIMIC-3: from DOB, capped at 89; eICU: ">89" → 89) |
| `gender_bin` | ✅ | ✅ | ✅ | ✅ | 0 = Female, 1 = Male |
| `insurance_cat` | ✅ | ✅ | — | — | 0=Medicare, 1=Medicaid, 2=Private, 3=Other |
| `admission_location_cat` | ✅ | ✅ | — | — | Label-encoded admission location |
| `ethnicity_cat` | — | — | ✅ | — | Label-encoded ethnicity |
| `bmi` | — | — | ✅ | — | Body mass index at admission (median-imputed, clipped 10–80) |
| `is_married` | — | — | — | ✅ | 1 if marital status = Married |
| `income` | — | — | — | ✅ | Annual household income (USD) |

### Admission Context

| Column | MIMIC-4 | MIMIC-3 | eICU | Synthea | Description |
|---|---|---|---|---|---|
| `is_emergency` | ✅ | ✅ | — | — | 1 if emergency/urgent admission |
| `unittype_cat` | — | — | ✅ | — | Label-encoded ICU unit type |
| `unitadmitsource_cat` | — | — | ✅ | — | Label-encoded ICU admit source |
| `total_cost` | — | — | — | ✅ | Total encounter cost (USD) |
| `admission_dayofweek` | ✅ | ✅ | — | ✅ | 0 = Monday … 6 = Sunday |
| `admission_month` | ✅ | ✅ | — | ✅ | 1–12 |
| `admission_year` | ✅ | ✅ | — | ✅ | Calendar year |
| `is_weekend` | ✅ | ✅ | — | ✅ | 1 if Saturday or Sunday |

### Elixhauser Comorbidities (31 binary flags)

All coded as 0/1. See Comorbidity Logic section above for derivation details.

| Column | Condition |
|---|---|
| `cm_chf` | Congestive Heart Failure |
| `cm_arrhythmia` | Cardiac Arrhythmias |
| `cm_valve` | Valvular Disease |
| `cm_pulm_circ` | Pulmonary Circulation Disorders |
| `cm_pvd` | Peripheral Vascular Disease |
| `cm_htn_uncomp` | Hypertension (Uncomplicated) |
| `cm_htn_comp` | Hypertension (Complicated) |
| `cm_paralysis` | Paralysis |
| `cm_neuro_other` | Other Neurological Disorders |
| `cm_copd` | Chronic Pulmonary Disease |
| `cm_dm_uncomp` | Diabetes (Uncomplicated) |
| `cm_dm_comp` | Diabetes (Complicated) |
| `cm_hypothyroid` | Hypothyroidism |
| `cm_renal` | Renal Failure |
| `cm_liver` | Liver Disease |
| `cm_ulcer` | Peptic Ulcer |
| `cm_hiv` | AIDS / HIV |
| `cm_lymphoma` | Lymphoma / Leukemia / Myeloma |
| `cm_metastatic` | Metastatic Cancer |
| `cm_solid_tumor` | Solid Tumor (without metastasis) |
| `cm_rheumatoid` | Rheumatoid / Collagen Vascular Disease |
| `cm_coagulopathy` | Coagulopathy |
| `cm_obesity` | Obesity |
| `cm_weight_loss` | Weight Loss / Malnutrition |
| `cm_fluid_electrolyte` | Fluid and Electrolyte Disorders |
| `cm_blood_loss` | Blood Loss Anemia |
| `cm_anemia` | Deficiency Anemia |
| `cm_alcohol` | Alcohol Abuse |
| `cm_drug` | Drug Abuse |
| `cm_psychosis` | Psychoses |
| `cm_depression` | Depression |

### Count Features (continuous, RobustScaled)

| Column | MIMIC-4 | MIMIC-3 | eICU | Synthea | Description |
|---|---|---|---|---|---|
| `num_diagnoses` | ✅ | ✅ | — | ✅ | Number of diagnosis codes per admission |
| `num_procedures` | ✅ | ✅ | — | ✅ | Number of procedure codes per admission |
| `num_transfers` | ✅ | ✅ | — | — | Number of ward/unit transfers |
| `num_prescriptions` | ✅ | ✅ | — | — | Number of medication orders |
| `num_labs` | ✅ | ✅ | ✅ | — | Number of lab tests ordered |
| `num_medications` | — | — | ✅ | — | Number of medication records (eICU) |
| `num_drugs` | — | — | — | ✅ | Active medications at time of admission (Synthea) |

### ICU & ED

| Column | MIMIC-4 | MIMIC-3 | eICU | Synthea | Description |
|---|---|---|---|---|---|
| `has_icu` | ✅ | ✅ | — | — | 1 if patient admitted to any ICU |
| `had_ed_visit` | ✅ | — | — | — | 1 if came through Emergency Department |
| `hospital_expire_flag` | ✅ | ✅ | ✅ | — | 1 if patient died during admission |

### eICU-only: APACHE Variables

Only available in `eicu_los.csv`. All prefixed with `apache_`. Missing values (originally `-1` sentinel in eICU) are converted to `NaN`; rows with any missing value are dropped. Seven columns with >50% missingness (blood gas and liver function tests, only collected in intubated/critically ill subsets) are removed entirely.

| Column | Type | Description |
|---|---|---|
| `apache_score` | continuous | Overall APACHE IV score |
| `apache_pred_mortality` | continuous | Predicted hospital mortality (%) |
| `apache_intubated` | binary | 1 if intubated at ICU admission |
| `apache_vent` | binary | 1 if on mechanical ventilation |
| `apache_dialysis` | binary | 1 if on dialysis |
| `apache_gcs_eyes` | continuous | GCS — eye response (1–4) |
| `apache_gcs_motor` | continuous | GCS — motor response (1–6) |
| `apache_gcs_verbal` | continuous | GCS — verbal response (1–5) |
| `apache_heartrate` | continuous | Heart rate (bpm) |
| `apache_meanbp` | continuous | Mean arterial blood pressure (mmHg) |
| `apache_temperature` | continuous | Body temperature (°C) |
| `apache_resprate` | continuous | Respiratory rate (breaths/min) |
| `apache_wbc` | continuous | White blood cell count (×10³/μL) |
| `apache_hematocrit` | continuous | Hematocrit (%) |
| `apache_sodium` | continuous | Serum sodium (mEq/L) |
| `apache_creatinine` | continuous | Serum creatinine (mg/dL) |
| `apache_glucose` | continuous | Blood glucose (mg/dL) |
| `apache_bun` | continuous | Blood urea nitrogen (mg/dL) |

### ED Triage Vitals — MIMIC-4 only

Recorded at ED triage. 0 for non-ED admissions. Not available in MIMIC-3 or Synthea Neo4j.

| Column | Description |
|---|---|
| `triage_temperature` | Body temperature (°F) |
| `triage_heartrate` | Heart rate (bpm) |
| `triage_resprate` | Respiratory rate (breaths/min) |
| `triage_o2sat` | Oxygen saturation (%) |
| `triage_sbp` | Systolic blood pressure (mmHg) |
| `triage_dbp` | Diastolic blood pressure (mmHg) |
| `triage_pain` | Pain score (0–10) |
| `triage_acuity` | ESI triage acuity level (1–5, lower = more urgent) |

---

## SHAP Interpretability

```python
import shap
import lightgbm as lgb
from dataset import load_mimic4_los

(X_train, y_train, X_val, y_val, X_test, y_test,
 feature_cols, continuous_cols, binary_cols, scaler) = load_mimic4_los(
     csv_path="processed/mimic4_los.csv"
 )

model = lgb.LGBMRegressor().fit(X_train, y_train)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# feature_cols matches X_test column order exactly — no reordering needed
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
```

`feature_cols` is always returned in the same order as the columns in `X_train/X_val/X_test` (continuous first, then binary), so it can be passed directly to SHAP without any reordering.
