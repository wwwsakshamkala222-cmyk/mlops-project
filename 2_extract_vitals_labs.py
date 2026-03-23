import pandas as pd
import os

print("Starting Phase 2: High-Efficiency Data Extraction...")

data_dir = './data'
cohort_file = os.path.join(data_dir, 'step1_lung_cohort_master.csv')
vitals_file = os.path.join(data_dir, 'chartevents.csv')
labs_file = os.path.join(data_dir, 'labevents.csv')

# 1. Load our suspect list (the 26 patients)
print("Loading master cohort...")
cohort = pd.read_csv(cohort_file)
patient_ids = cohort['subject_id'].unique()
print(f"Tracking {len(patient_ids)} patients...")

# 2. Extract Vitals using Chunking (Memory Safe)
print("Sifting through massive Vitals file (chartevents.csv)... this might take a minute.")
vitals_chunks = []
# We read the file 100,000 rows at a time
for chunk in pd.read_csv(vitals_file, chunksize=100000, low_memory=False):
    # Only keep rows belonging to our specific patients
    filtered_chunk = chunk[chunk['subject_id'].isin(patient_ids)]
    vitals_chunks.append(filtered_chunk)

# Combine all the filtered chunks into one neat dataframe
patient_vitals = pd.concat(vitals_chunks)
vitals_output = os.path.join(data_dir, 'step2_vitals.csv')
patient_vitals.to_csv(vitals_output, index=False)
print(f"SUCCESS! Extracted {len(patient_vitals)} vital sign records. Saved to: {vitals_output}")

# 3. Extract Labs using Chunking
print("Sifting through Labs file (labevents.csv)...")
labs_chunks = []
for chunk in pd.read_csv(labs_file, chunksize=100000, low_memory=False):
    filtered_chunk = chunk[chunk['subject_id'].isin(patient_ids)]
    labs_chunks.append(filtered_chunk)

patient_labs = pd.concat(labs_chunks)
labs_output = os.path.join(data_dir, 'step2_labs.csv')
patient_labs.to_csv(labs_output, index=False)
print(f"SUCCESS! Extracted {len(patient_labs)} lab records. Saved to: {labs_output}")

print("Phase 2 Complete. All necessary data has been isolated.")