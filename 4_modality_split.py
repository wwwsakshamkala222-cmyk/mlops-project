import pandas as pd
import numpy as np
import os
import torch

print("Starting Phase 4: Modality Split & Tensor Creation...")

data_dir = './data'
cohort_file = os.path.join(data_dir, 'step1_lung_cohort_master.csv')
vitals_file = os.path.join(data_dir, 'step3_vitals_normalized.csv')
labs_file = os.path.join(data_dir, 'step3_labs_normalized.csv')

# 1. Load the pristine data
print("Loading preprocessed data...")
cohort = pd.read_csv(cohort_file)
vitals = pd.read_csv(vitals_file)
labs = pd.read_csv(labs_file)

# 2. Prepare Modality A (Static Data: Age, Gender)
print("Processing Modality A (Static Data)...")
# Convert Gender to numbers (F=1, M=0) so the AI can process it
cohort['gender_num'] = cohort['gender'].apply(lambda x: 1 if x == 'F' else 0)
static_features = cohort[['subject_id', 'anchor_age', 'gender_num']].drop_duplicates('subject_id')

# Create a fast-lookup dictionary for static info
static_dict = static_features.set_index('subject_id').to_dict('index')

# 3. Prepare Modality B (Time-Series Data)
print("Processing Modality B (Time-Series Data)...")
# Combine vitals and labs into one giant timeline, keeping only what the AI needs
timeline = pd.concat([
    vitals[['subject_id', 'charttime', 'itemid', 'normalized_value']],
    labs[['subject_id', 'charttime', 'itemid', 'normalized_value']]
])

# Sort everything chronologically per patient
timeline['charttime'] = pd.to_datetime(timeline['charttime'])
timeline = timeline.sort_values(by=['subject_id', 'charttime'])

# 4. Building the PyTorch Matrices (Tensors)
# A Transformer requires a fixed timeline length. We will take the first 50 events.
MAX_EVENTS = 50
patient_ids = cohort['subject_id'].unique()

static_tensors = []
temporal_tensors = []

print("Structuring PyTorch Tensors (This creates the 3D matrices for the GPU)...")
for pid in patient_ids:
    # --- Build Modality A (Static) ---
    p_static = static_dict.get(pid, {'anchor_age': 0, 'gender_num': 0})
    static_tensors.append([p_static['anchor_age'], p_static['gender_num']])
    
    # --- Build Modality B (Temporal) ---
    p_timeline = timeline[timeline['subject_id'] == pid].head(MAX_EVENTS)
    
    vals = p_timeline['normalized_value'].values
    items = p_timeline['itemid'].values
    
    # Create an empty matrix of zeros: [50 time steps, 2 features (ID and Value)]
    patient_temporal = np.zeros((MAX_EVENTS, 2))
    
    # Fill the matrix with the patient's actual chronological events
    for i in range(len(vals)):
        patient_temporal[i, 0] = items[i] / 1000000.0  # Scale down the medical ID number
        patient_temporal[i, 1] = vals[i]               # The 0-1 normalized medical value
        
    temporal_tensors.append(patient_temporal)

# 5. Convert lists to highly optimized PyTorch Tensors
tensor_A = torch.tensor(static_tensors, dtype=torch.float32)
tensor_B = torch.tensor(temporal_tensors, dtype=torch.float32)

# 6. Save the tensors directly to the hard drive
out_A = os.path.join(data_dir, 'modality_A_static.pt')
out_B = os.path.join(data_dir, 'modality_B_temporal.pt')

torch.save(tensor_A, out_A)
torch.save(tensor_B, out_B)

print("\nSUCCESS! AI Inputs generated and saved.")
print(f"Modality A (Static) saved to: {out_A} | Matrix Shape: {tensor_A.shape}")
print(f"Modality B (Temporal) saved to: {out_B} | Matrix Shape: {tensor_B.shape}")