import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

print("Starting Phase 3: Missing Values & Min-Max Normalization...")

data_dir = './data'
vitals_file = os.path.join(data_dir, 'step2_vitals.csv')
labs_file = os.path.join(data_dir, 'step2_labs.csv')

# 1. Load the extracted data
print("Loading Vitals and Labs...")
vitals = pd.read_csv(vitals_file)
labs = pd.read_csv(labs_file)

# 2. Focus on the actual numerical values we care about
# In MIMIC, 'valuenum' is the column that holds the actual number (like 98.6 for temp)
# We will drop rows where there is no numerical value at all
print("Cleaning out empty records...")
vitals_clean = vitals.dropna(subset=['valuenum']).copy()
labs_clean = labs.dropna(subset=['valuenum']).copy()

# 3. Handling Missing Data (Imputation)
# If a value is still missing somehow, we fill it with the 'median' (average) of that column
# This prevents the AI from crashing when it hits a blank space
print("Filling in the blanks (Imputation)...")
vitals_clean['valuenum'] = vitals_clean['valuenum'].fillna(vitals_clean['valuenum'].median())
labs_clean['valuenum'] = labs_clean['valuenum'].fillna(labs_clean['valuenum'].median())

# 4. The Min-Max Normalization
# This shrinks every number in the 'valuenum' column to be exactly between 0 and 1
print("Applying Min-Max Normalization (Scaling down to 0-1)...")
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale Vitals
vitals_clean['normalized_value'] = scaler.fit_transform(vitals_clean[['valuenum']])

# Scale Labs
labs_clean['normalized_value'] = scaler.fit_transform(labs_clean[['valuenum']])

# 5. Save the perfectly formatted data
vitals_output = os.path.join(data_dir, 'step3_vitals_normalized.csv')
labs_output = os.path.join(data_dir, 'step3_labs_normalized.csv')

vitals_clean.to_csv(vitals_output, index=False)
labs_clean.to_csv(labs_output, index=False)

print(f"SUCCESS! Data normalized and saved to:")
print(f"- {vitals_output}")
print(f"- {labs_output}")
print("The clues are now in a language the AI can understand.")