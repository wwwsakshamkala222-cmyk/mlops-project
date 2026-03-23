import pandas as pd
import os

print("Starting Phase 1: MedEdge-Ops Lung Disease Cohort Extraction...")

# 1. Define the file paths (pointing to your data folder)
data_dir = './data'
diagnoses_file = os.path.join(data_dir, 'diagnoses_icd.csv')
patients_file = os.path.join(data_dir, 'patients.csv')
icustays_file = os.path.join(data_dir, 'icustays.csv')

# 2. Define the exact ICD-9 billing codes for Lung Diseases
# 493.x = Asthma | 491.x, 492.x, 496.x = COPD | 480.x - 486.x = Pneumonia
lung_codes = ('493', '491', '492', '496', '480', '481', '482', '483', '484', '485', '486')

# 3. Load and filter the Diagnoses table
print("Loading diagnoses...")
diagnoses = pd.read_csv(diagnoses_file)
# Drop rows where the code is missing to prevent errors
diagnoses = diagnoses.dropna(subset=['icd_code'])
# Find all rows where the code starts with our lung disease numbers
lung_diagnoses = diagnoses[diagnoses['icd_code'].astype(str).str.startswith(lung_codes)]

# Extract just the unique patient IDs (subject_id)
lung_patient_ids = lung_diagnoses['subject_id'].unique()
print(f"Found {len(lung_patient_ids)} unique patients with lung diseases.")

# 4. Load the Patients table and keep ONLY our lung patients
print("Filtering patient demographics...")
patients = pd.read_csv(patients_file)
lung_patients = patients[patients['subject_id'].isin(lung_patient_ids)]

# 5. Load the ICU Stays table and keep ONLY our lung patients
# (We need this to know exactly when they entered and left the ICU)
print("Filtering ICU stay records...")
icustays = pd.read_csv(icustays_file)
lung_icustays = icustays[icustays['subject_id'].isin(lung_patient_ids)]

# 6. Merge Patients and ICU Stays together into one master "Static Data" table
# This gives us their Age/Gender matched directly with their ICU admission times
print("Merging data into master cohort file...")
master_cohort = pd.merge(lung_patients, lung_icustays, on='subject_id', how='inner')

# 7. Save this highly filtered, lightweight data to a new CSV
output_file = os.path.join(data_dir, 'step1_lung_cohort_master.csv')
master_cohort.to_csv(output_file, index=False)

print(f"SUCCESS! Master cohort saved to: {output_file}")
print("Your computer's memory is safe, and the suspect list is ready.")