import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict
from collections import Counter

INPUT_DIR = "data/ECG/"
TRAIN_DIR = "split_data/ECG/training/"
TEST_DIR = "split_data/ECG/testing/"
SPLIT_RATIO = 0.2

# Create directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def extract_label(hea_path):
    """Extract disease class label from header file."""
    with open(hea_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") and "Reason for admission:" in line:
                parts = line.split("Reason for admission:")[-1].strip().split(",")
                return parts[0].strip()
    return None

def find_record_files(base_dir):
    """Find all unique record base names (without extension)."""
    dat_files = glob.glob(os.path.join(base_dir, "**/*.dat"), recursive=True)
    return [os.path.splitext(f)[0] for f in dat_files]

def copy_record(record_base, destination_dir):
    """Copy all files (.hea, .dat, .atr, etc.) related to a record."""
    extensions = ['.hea', '.dat', '.atr', '.xml']
    for ext in extensions:
        src = record_base + ext
        if os.path.exists(src):
            shutil.copy(src, os.path.join(destination_dir, os.path.basename(src)))

def stratified_split(records, labels, test_size=0.2):
    """Split records into stratified train/test sets."""
    return train_test_split(records, test_size=test_size, stratify=labels, random_state=42)

def main():
    record_bases = find_record_files(INPUT_DIR)
    print(f"Found {len(record_bases)} ECG records.")

    records = []
    labels = []

    for record_base in record_bases:
        hea_path = record_base + ".hea"
        label = extract_label(hea_path)
        if label:
            records.append(record_base)
            labels.append(label)

    print(f"{len(records)} records have usable labels.")

    label_counts = Counter(labels)
    filtered_records = []
    filtered_labels = []

    for r, l in zip(records, labels):
        if label_counts[l] >= 2:
            filtered_records.append(r)
            filtered_labels.append(l)

    print(f"{len(filtered_records)} records after filtering labels with at least 2 samples.")

    if len(filtered_labels) == 0:
        print("❌ No valid labels with at least 2 samples. Aborting.")
        return

    train_records, test_records = stratified_split(filtered_records, filtered_labels, test_size=SPLIT_RATIO)

    print(f"✅ Splitting {len(records)} ECG records → {len(train_records)} train / {len(test_records)} test")

    for record in train_records:
        copy_record(record, TRAIN_DIR)

    for record in test_records:
        copy_record(record, TEST_DIR)

    print("✅ ECG stratified split completed.")

if __name__ == "__main__":
    main()
