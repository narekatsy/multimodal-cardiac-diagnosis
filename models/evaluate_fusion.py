import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from datasets import MRIDataset, ECGDataset
from mri_encoder import MRIEncoder
from ecg_encoder import ECGEncoder
from metadata_encoder import MetadataEncoder
from multimodal_fusion_transformer import MultimodalFusionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
num_classes = 11  # Fusion output classes

# Load models
mri_encoder = MRIEncoder().to(device)
ecg_encoder = ECGEncoder().to(device)
mri_meta_encoder = MetadataEncoder(input_dim=5).to(device)
ecg_meta_encoder = MetadataEncoder(input_dim=4).to(device)
fusion_model = MultimodalFusionTransformer(embed_dim=512, num_heads=4, num_layers=2, num_classes=num_classes).to(device)

# Load weights
mri_encoder.load_state_dict(torch.load('checkpoints/pretrained_mri_encoder.pt'))
ecg_encoder.load_state_dict(torch.load('checkpoints/pretrained_ecg_encoder.pt'))
fusion_model.load_state_dict(torch.load('checkpoints/fusion_model.pt'))

# Set to eval
mri_encoder.eval()
ecg_encoder.eval()
mri_meta_encoder.eval()
ecg_meta_encoder.eval()
fusion_model.eval()

# Select modality to evaluate
modality = "mri"  # or "ecg"
if modality == "mri":
    dataset = MRIDataset(
        scan_dir='processed_data/mri/testing/scans',
        meta_dir='processed_data/mri/testing/metadata',
        label_dir='processed_data/mri/testing/labels'
    )
else:
    dataset = ECGDataset(
        signal_dir='processed_data/ecg/testing/signals',
        meta_dir='processed_data/ecg/testing/metadata',
        label_dir='processed_data/ecg/testing/labels'
    )

loader = DataLoader(dataset, batch_size=batch_size)

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for x, meta, y in loader:
        x, meta, y = x.to(device), meta.to(device), y.to(device)

        if modality == "mri":
            emb = mri_encoder(x)
            meta_emb = mri_meta_encoder(meta)
            output = fusion_model(mri_emb=emb, ecg_emb=None, meta_emb=meta_emb)
        else:
            emb = ecg_encoder(x)
            meta_emb = ecg_meta_encoder(meta)
            output = fusion_model(mri_emb=None, ecg_emb=emb, meta_emb=meta_emb)

        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Report
print("\nFUSION MODEL -", modality.upper())
print(classification_report(all_labels, all_preds, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
