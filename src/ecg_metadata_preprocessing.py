import os
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

input_dir = "data/ECG/"
output_dir = "processed_data/metadata/ecg/"
os.makedirs(output_dir, exist_ok=True)

all_metadata = []
usable_entries = []

for hea_file in glob.glob(os.path.join(input_dir, "**/*.hea"), recursive=True):
    with open(hea_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        patient_meta = {"patient_id": os.path.basename(hea_file).split(".")[0]}

        # Parse metadata from lines starting with #
        for line in lines:
            if line.startswith("#"):
                line = line.strip("# ").strip()
                if line.startswith("age:"):
                    try:
                        patient_meta["age"] = int(line.split(":", 1)[1].strip())
                    except:
                        continue
                elif line.startswith("sex:"):
                    patient_meta["sex"] = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Smoker:"):
                    patient_meta["smoker"] = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Number of coronary vessels involved:"):
                    val = line.split(":")[1].strip()
                    try:
                        patient_meta["vessels_involved"] = int(val)
                    except:
                        patient_meta["vessels_involved"] = 0  # fallback

        # Check if minimum fields are present
        if all(k in patient_meta for k in ["age", "sex", "smoker", "vessels_involved"]):
            all_metadata.append(patient_meta)

# Encode categorical variables
sex_encoder = LabelEncoder()
smoker_encoder = LabelEncoder()

sex_encoded = sex_encoder.fit_transform([x["sex"] for x in all_metadata])
smoker_encoded = smoker_encoder.fit_transform([x["smoker"] for x in all_metadata])

for idx, entry in enumerate(all_metadata):
    entry["sex"] = sex_encoded[idx]
    entry["smoker"] = smoker_encoded[idx]

# Normalize continuous features
scaler = MinMaxScaler()
age_vessels = np.array([[x["age"], x["vessels_involved"]] for x in all_metadata])
age_vessels_scaled = scaler.fit_transform(age_vessels)

for i, entry in enumerate(all_metadata):
    entry["age"] = age_vessels_scaled[i][0]
    entry["vessels_involved"] = age_vessels_scaled[i][1]

# Save per patient
for entry in all_metadata:
    vec = np.array([
        entry["age"],
        entry["sex"],
        entry["smoker"],
        entry["vessels_involved"]
    ], dtype=np.float32)
    
    out_path = os.path.join(output_dir, f"{entry['patient_id']}.npy")
    np.save(out_path, vec)

print("✅ ECG metadata preprocessing complete.")
