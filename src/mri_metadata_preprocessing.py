import os
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

input_dir = "split_data/MRI/training/"
output_dir = "processed_data/mri/training/metadata/"
os.makedirs(output_dir, exist_ok=True)

all_metadata = []
group_labels = []

for info_file in glob.glob(os.path.join(input_dir, "*/info.cfg")):
    patient_data = {}
    with open(info_file, "r") as f:
        for line in f:
            key, value = line.strip().split(":")
            key, value = key.strip(), value.strip()
            if key in ["ED", "ES", "NbFrame"]:
                patient_data[key] = int(value)
            elif key in ["Height", "Weight"]:
                patient_data[key] = float(value)
            elif key == "Group":
                patient_data[key] = value
                group_labels.append(value)
    patient_data["patient_id"] = os.path.basename(os.path.dirname(info_file))
    all_metadata.append(patient_data)

label_encoder = LabelEncoder()
group_encoded = label_encoder.fit_transform([x["Group"] for x in all_metadata])
for idx, entry in enumerate(all_metadata):
    entry["Group"] = group_encoded[idx]

scaler = MinMaxScaler()
height_weight = np.array([[x["Height"], x["Weight"]] for x in all_metadata])
height_weight_scaled = scaler.fit_transform(height_weight)

for i, entry in enumerate(all_metadata):
    entry["Height"] = height_weight_scaled[i][0]
    entry["Weight"] = height_weight_scaled[i][1]

# Save per patient
metadata_output_dir = "processed_data/mri/training/metadata/"
label_output_dir = "processed_data/mri/training/labels/"
os.makedirs(metadata_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)

for entry in all_metadata:
    metadata_vec = np.array([
        entry["ED"],
        entry["ES"],
        entry["NbFrame"],
        entry["Height"],
        entry["Weight"]
    ], dtype=np.float32)
    label_vec = np.array(entry["Group"], dtype=np.int64)

    patient_id = entry['patient_id']
    np.save(os.path.join(metadata_output_dir, f"{patient_id}.npy"), metadata_vec)
    np.save(os.path.join(label_output_dir, f"{patient_id}.npy"), label_vec)


print("✅ MRI metadata preprocessing complete.")
