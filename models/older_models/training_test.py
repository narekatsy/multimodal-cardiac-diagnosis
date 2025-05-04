import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datasets import MRIDataset, ECGDataset
from mri_encoder import MRIEncoder
from ecg_encoder import ECGEncoder
from metadata_encoder import MetadataEncoder
from multimodal_fusion_transformer import MultimodalFusionTransformer
from torch.utils.data import RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 2.1: Instantiate Datasets ──────────────────────────────────────────────
mri_ds = MRIDataset(
    scan_dir  ="processed_data/mri/training/scans",
    meta_dir  ="processed_data/mri/training/metadata",
    label_dir ="processed_data/mri/training/labels"
)
ecg_ds = ECGDataset(
    signal_dir="processed_data/ecg/training/signals",
    meta_dir  ="processed_data/ecg/training/metadata",
    label_dir ="processed_data/ecg/training/labels"
)

# Take only 5 samples each
mri_ds = Subset(mri_ds, list(range(min(5, len(mri_ds)))))
ecg_ds = Subset(ecg_ds, list(range(min(5, len(ecg_ds)))))

mri_loader = DataLoader(mri_ds, batch_size=5, sampler=RandomSampler(mri_ds))
ecg_loader = DataLoader(ecg_ds, batch_size=5, sampler=RandomSampler(ecg_ds))

# ─── 2.2: Pull one batch ────────────────────────────────────────────────────
mri_scan, mri_meta, mri_label = next(iter(mri_loader))
ecg_sig,  ecg_meta, ecg_label  = next(iter(ecg_loader))

mri_scan, mri_meta, mri_label = mri_scan.to(device), mri_meta.to(device), mri_label.to(device)
ecg_sig,  ecg_meta,  ecg_label  = ecg_sig.to(device), ecg_meta.to(device),  ecg_label.to(device)

# ─── 2.3: Build Models ─────────────────────────────────────────────────────
mri_meta = mri_meta[:, :4]

if mri_scan.ndim == 4:  # (B, D, H, W)
    mri_scan = mri_scan.unsqueeze(1)
elif mri_scan.ndim == 5 and mri_scan.shape[1] != 1:
    mri_scan = mri_scan.permute(0, 2, 1, 3, 4)  # (B, D, 1, H, W) → (B, 1, D, H, W)

if ecg_sig.ndim == 3 and ecg_sig.shape[-1] == 1:
    ecg_sig = ecg_sig.squeeze(-1)  # (B, 115200)
if ecg_sig.shape[1] == 115200:
    ecg_sig = ecg_sig.view(-1, 15, 7680)  # Example: reshape assuming T=7680

# Check for NaNs or constant values
if torch.any(torch.isnan(ecg_sig)):
    print(f"NaNs found in ECG signal")
if torch.all(ecg_sig == 0):
    print(f"All zeros found in ECG signal")

mri_enc  = MRIEncoder().to(device)
ecg_enc  = ECGEncoder().to(device)
meta_enc = MetadataEncoder(input_dim=mri_meta.shape[1]).to(device)  # same for ECG
fusion   = MultimodalFusionTransformer(
    embed_dim=512,
    num_heads=4,
    num_layers=2,
    num_classes=max(mri_label.max().item(), ecg_label.max().item()) + 1
).to(device)

# ─── 2.4: Simple Classifier Heads for Encoder-only Pretraining ─────────────
mri_clf = nn.Linear(512, mri_label.max().item() + 1).to(device)
ecg_clf = nn.Linear(512, ecg_label.max().item() + 1).to(device)

# Loss + optim
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(
    list(mri_enc.parameters()) + list(mri_clf.parameters()) +
    list(ecg_enc.parameters()) + list(ecg_clf.parameters()) +
    list(meta_enc.parameters()) + list(fusion.parameters()),
    lr=0.001
)

# ─── 2.5: Forward & Backward ────────────────────────────────────────────────
# MRI‐only step
mri_emb = mri_enc(mri_scan)            # (5,512)
mri_out = mri_clf(mri_emb)             # (5, num_mri_classes)
mri_loss = criterion(mri_out, mri_label)

# ECG‐only step
ecg_emb = ecg_enc(ecg_sig)             # (5,512)
ecg_out = ecg_clf(ecg_emb)             # (5, num_ecg_classes)
ecg_loss = criterion(ecg_out, ecg_label)

# Fusion step (use all 3 embeddings; here we pretend MRI+ECG+meta all present)
# Note: if you want missing modality test, set mri_emb=None or ecg_emb=None
fusion_emb_mri  = mri_emb
fusion_emb_ecg  = ecg_emb
fusion_emb_meta = meta_enc(ecg_meta)  # or mri_meta, shapes match

fusion_out = fusion(fusion_emb_mri, fusion_emb_ecg, fusion_emb_meta)
# To keep it simple, train on ECG label if classes overlap; otherwise skip
fusion_loss = criterion(fusion_out, ecg_label)

total_loss = mri_loss + ecg_loss + fusion_loss

# Forward pass
ecg_output = ecg_enc(ecg_sig)
print("ECG model output:", ecg_output[:2])
target = torch.zeros_like(ecg_output)
loss = torch.nn.MSELoss()(ecg_output, target)
print("ECG MSE loss:", loss.item())

opt.zero_grad()
total_loss.backward()
opt.step()

print(f"✅ MRI loss: {mri_loss.item():.4f}")
print(f"✅ ECG loss: {ecg_loss.item()}")
print(f"✅ Fusion loss: {fusion_loss.item():.4f}")
