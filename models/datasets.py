import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter, defaultdict

def compute_label_mapping(label_dir, min_samples_per_class=8):
    """Generates a mapping from original labels to merged labels if sample count is low."""
    label_counts = Counter()
    label_map = {}
    
    for f in os.listdir(label_dir):
        if f.endswith('.npy'):
            label = int(np.load(os.path.join(label_dir, f)))
            label_counts[label] += 1

    # Assign a new label to underrepresented classes
    merged_class_id = max(label_counts.keys()) + 1
    for label, count in label_counts.items():
        if count < min_samples_per_class:
            label_map[label] = merged_class_id
        else:
            label_map[label] = label

    return label_map


class MRIDataset(Dataset):
    def __init__(self, scan_dir, meta_dir, label_dir, transform=None, min_samples_per_class=8):
        self.scan_dir = scan_dir
        self.meta_dir = meta_dir
        self.label_dir = label_dir
        self.transform = transform

        scans = sorted(f for f in os.listdir(scan_dir) if f.endswith('.npy'))
        metas = set(os.listdir(meta_dir))
        labels = set(os.listdir(label_dir))

        self.samples = [f for f in scans if f in metas and f in labels]
        self.label_map = compute_label_mapping(label_dir, min_samples_per_class)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn = self.samples[idx]
        scan = np.load(os.path.join(self.scan_dir, fn))  # (1, D, H, W)
        meta = np.load(os.path.join(self.meta_dir, fn))  # (F_meta,)
        label = int(np.load(os.path.join(self.label_dir, fn)))
        label = self.label_map[label]

        if scan.ndim == 4:
            scan = np.transpose(scan, (3, 0, 1, 2))  # (T, D, H, W)
        if scan.shape[0] != 1:
            scan = np.expand_dims(scan, axis=0)  # (1, D, H, W)

        scan = torch.tensor(scan, dtype=torch.float32)
        meta = torch.tensor(meta, dtype=torch.float32)

        print("[Status] MRI scan in batch")

        if self.transform:
            scan = self.transform(scan)

        return scan, meta, label


class ECGDataset(Dataset):
    def __init__(self, signal_dir, meta_dir, label_dir, transform=None, min_samples_per_class=8):
        self.signal_dir = signal_dir
        self.meta_dir = meta_dir
        self.label_dir = label_dir
        self.transform = transform

        signals = sorted(f for f in os.listdir(signal_dir) if f.endswith('.npy'))
        metas = set(os.listdir(meta_dir))
        labels = set(os.listdir(label_dir))

        self.samples = [f for f in signals if f in metas and f in labels]
        self.label_map = compute_label_mapping(label_dir, min_samples_per_class)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn = self.samples[idx]
        sig = np.load(os.path.join(self.signal_dir, fn))  # Expected shape: (15, T)

        print(f"[DEBUG] ECG shape before truncation: {sig.shape}")  # Debugging line

        if sig.ndim == 2 and sig.shape[1] == 1:
            sig = sig.squeeze(1)
        if sig.ndim == 1:
            remainder = sig.shape[0] % 15
            if remainder != 0:
                print(f"[WARNING] Signal length {sig.shape[0]} not divisible by 15. Truncating remainder.")
                sig = sig[:sig.shape[0] - remainder]
            sig = sig.reshape(15, -1)
        elif sig.shape[0] != 15:
            raise ValueError(f"Unexpected shape for ECG signal: {sig.shape}")

        fixed_length = 30000
        if sig.shape[1] > fixed_length:
            sig = sig[:, :fixed_length]
        else:
            pad_width = fixed_length - sig.shape[1]
            sig = np.pad(sig, ((0, 0), (0, pad_width)), mode='constant')

        meta = np.load(os.path.join(self.meta_dir, fn))
        label = int(np.load(os.path.join(self.label_dir, fn)))
        label = self.label_map[label]

        sig = torch.tensor(sig, dtype=torch.float32)
        meta = torch.tensor(meta, dtype=torch.float32)

        if torch.any(torch.isnan(sig)):
            print(f"NaNs found in ECG signal at index {idx}, file {fn}")
        if torch.all(sig == 0):
            print(f"All zeros found in ECG signal at index {idx}, file {fn}")

        if self.transform:
            sig = self.transform(sig)

        return sig, meta, label
