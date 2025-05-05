import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import MRIDataset, ECGDataset
from mri_encoder import MRIEncoder
from ecg_encoder import ECGEncoder
from metadata_encoder import MetadataEncoder
from multimodal_fusion_transformer import MultimodalFusionTransformer

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 8
num_epochs = 30
learning_rate = 1e-4
num_classes = 11

# Datasets
mri_dataset = MRIDataset(
    scan_dir='processed_data/mri/training/scans',
    meta_dir='processed_data/mri/training/metadata',
    label_dir='processed_data/mri/training/labels'
)
ecg_dataset = ECGDataset(
    signal_dir='processed_data/ecg/training/signals',
    meta_dir='processed_data/ecg/training/metadata',
    label_dir='processed_data/ecg/training/labels'
)

mri_loader = DataLoader(mri_dataset, batch_size=batch_size, shuffle=True)
ecg_loader = DataLoader(ecg_dataset, batch_size=batch_size, shuffle=True)

# Models
mri_encoder = MRIEncoder().to(device)
ecg_encoder = ECGEncoder().to(device)
mri_meta_encoder = MetadataEncoder(input_dim=5).to(device)
ecg_meta_encoder = MetadataEncoder(input_dim=4).to(device)
fusion_model = MultimodalFusionTransformer(embed_dim=512, num_heads=4, num_layers=2, num_classes=num_classes).to(device)

# Load pretrained weights
mri_encoder.load_state_dict(torch.load('checkpoints/pretrained_mri_encoder.pt'))
ecg_encoder.load_state_dict(torch.load('checkpoints/pretrained_ecg_encoder.pt'))

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(fusion_model.parameters()) +
    list(mri_encoder.parameters()) +
    list(ecg_encoder.parameters()),
    lr=learning_rate,
    weight_decay=1e-5  # L2 reg
)

# Training Loop
for epoch in range(num_epochs):
    fusion_model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    #### --- Train on MRI Batches ---
    for scans, metas, labels in mri_loader:
        scans = scans.to(device)
        metas = metas.to(device)
        labels = labels.to(device)

        mri_emb = mri_encoder(scans)
        mri_meta_emb = mri_meta_encoder(metas)

        outputs = fusion_model(mri_emb=mri_emb, ecg_emb=None, meta_emb=mri_meta_emb)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

    #### --- Train on ECG Batches ---
    for signals, metas, labels in ecg_loader:
        signals = signals.to(device)
        metas = metas.to(device)
        labels = labels.to(device)

        ecg_emb = ecg_encoder(signals)
        ecg_meta_emb = ecg_meta_encoder(metas)

        outputs = fusion_model(mri_emb=None, ecg_emb=ecg_emb, meta_emb=ecg_meta_emb)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

    acc = total_correct / total_samples
    print(f"Epoch {epoch+1}/{num_epochs} | Total Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")

# Save the model
torch.save(fusion_model.state_dict(), 'checkpoints/fusion_model.pt')
