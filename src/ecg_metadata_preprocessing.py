import os
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

input_dir = "split_data/ECG/training/"
output_dir = "processed_data/ecg/training/metadata/"
os.makedirs(output_dir, exist_ok=True)

all_entries = []

DIAGNOSIS_MAP = {
    "Myocardial infarction": 0,
    "Cardiomyopathy": 1,
    "Heart failure (NYHA 2)": 1,
    "Heart failure (NYHA 3)": 1,
    "Heart failure (NYHA 4)": 1,
    "Bundle branch block": 2,
    "Dysrhythmia": 3,
    "Hypertrophy": 4,
    "Valvular heart disease": 5,
    "Myocarditis": 6,
    "Palpitation": 7,
    "Stable angina": 7,
    "Unstable angina": 7,
    "Miscellaneous": 7,
    "Healthy control": 8,
}

for hea_path in glob.glob(os.path.join(input_dir, "**/*.hea"), recursive=True):
    patient_id = os.path.basename(os.path.dirname(hea_path))
    record_id = os.path.splitext(os.path.basename(hea_path))[0]

    metadata = {"patient_id": patient_id, "record_id": record_id}

    with open(hea_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                line = line.strip("# ").strip()
                if line.startswith("age:"):
                    try:
                        metadata["age"] = int(line.split(":", 1)[1].strip())
                    except:
                        continue
                elif line.startswith("sex:"):
                    metadata["sex"] = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Smoker:"):
                    metadata["smoker"] = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Number of coronary vessels involved:"):
                    try:
                        metadata["vessels_involved"] = int(line.split(":", 1)[1].strip())
                    except:
                        metadata["vessels_involved"] = 0

    if all(k in metadata for k in ["age", "sex", "smoker", "vessels_involved"]):
        all_entries.append(metadata)

# Encode categorical values
sex_enc = LabelEncoder()
smoker_enc = LabelEncoder()

sex_encoded = sex_enc.fit_transform([x["sex"] for x in all_entries])
smoker_encoded = smoker_enc.fit_transform([x["smoker"] for x in all_entries])

# Normalize continuous values
scaler = MinMaxScaler()
age_vessels = np.array([[x["age"], x["vessels_involved"]] for x in all_entries])
age_vessels_scaled = scaler.fit_transform(age_vessels)

# Final output
metadata_output_dir = "processed_data/ecg/training/metadata/"
label_output_dir = "processed_data/ecg/training/labels/"
os.makedirs(metadata_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)

for i, entry in enumerate(all_entries):
    diagnosis_label = None
    hea_path = os.path.join(input_dir, entry["patient_id"], entry["record_id"] + ".hea")
    with open(hea_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# Reason for admission:"):
                reason = line.strip().split(":")[-1].strip()
                for key in DIAGNOSIS_MAP:
                    if key.lower() in reason.lower():
                        diagnosis_label = DIAGNOSIS_MAP[key]
                        break
                break

    if diagnosis_label is None:
        continue

    vec = np.array([
        age_vessels_scaled[i][0],
        sex_encoded[i],
        smoker_encoded[i],
        age_vessels_scaled[i][1]
    ], dtype=np.float32)

    filename = f"{entry['patient_id']}_{entry['record_id']}.npy"
    np.save(os.path.join(metadata_output_dir, filename), vec)
    np.save(os.path.join(label_output_dir, filename), np.array(diagnosis_label, dtype=np.int64))


print("✅ ECG metadata preprocessing complete.")