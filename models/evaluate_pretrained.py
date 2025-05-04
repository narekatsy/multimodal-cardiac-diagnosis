import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from ecg_encoder import ECGEncoder
from mri_encoder import MRIEncoder
from datasets import ECGDataset, MRIDataset

# Settings (change based on which model you're evaluating)
modality = "ecg"  # or "mri"
num_classes = 9 if modality == "ecg" else 5
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and data
if modality == "ecg":
    dataset = ECGDataset(
        signal_dir='processed_data/ecg/testing/signals',
        meta_dir='processed_data/ecg/testing/metadata',
        label_dir='processed_data/ecg/testing/labels'
    )
    encoder = ECGEncoder()
    model = nn.Sequential(encoder, nn.Linear(512, num_classes))
    model.load_state_dict(torch.load("checkpoints/pretrained_ecg_encoder.pt"), strict=False)
else:
    dataset = MRIDataset(
        scan_dir='processed_data/mri/testing/scans',
        meta_dir='processed_data/mri/testing/metadata',
        label_dir='processed_data/mri/testing/labels'
    )
    encoder = MRIEncoder()
    model = nn.Sequential(encoder, nn.Linear(512, num_classes))
    model.load_state_dict(torch.load("checkpoints/pretrained_mri_encoder.pt"), strict=False)

model = model.to(device)
model.eval()
loader = DataLoader(dataset, batch_size=batch_size)

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for data in loader:
        x, _, y = data
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Metrics
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
