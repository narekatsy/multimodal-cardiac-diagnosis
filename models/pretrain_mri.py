import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mri_encoder import MRIEncoder
from datasets import MRIDataset
import torch.optim as optim

# Load Dataset
mri_ds = MRIDataset(scan_dir='processed_data/mri/training/scans',
                    meta_dir='processed_data/mri/training/metadata',
                    label_dir='processed_data/mri/training/labels')
mri_loader = DataLoader(mri_ds, batch_size=8, shuffle=True)

# Initialize model
encoder = MRIEncoder(embed_dim=512)
classifier = nn.Linear(512, 5)  # 5 classes
model = nn.Sequential(encoder, classifier)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    for scan, meta, label in mri_loader:
        scan, label = scan.to(device), label.to(device)

        optimizer.zero_grad()
        out = model(scan)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == label).sum().item()

    acc = correct / len(mri_loader.dataset)
    print(f"Epoch {epoch+1} | MRI Loss: {total_loss:.4f} | Acc: {acc:.4f}")

    torch.save(encoder.state_dict(), "checkpoints/pretrained_mri_encoder.pt")

