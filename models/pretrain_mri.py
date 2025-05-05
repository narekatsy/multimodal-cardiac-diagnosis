import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from mri_encoder import MRIEncoder
from datasets import MRIDataset
import torch.optim as optim

# Load Dataset
mri_ds = MRIDataset(scan_dir='processed_data/mri/training/scans',
                    meta_dir='processed_data/mri/training/metadata',
                    label_dir='processed_data/mri/training/labels')
train_size = int(0.8 * len(mri_ds))
val_size = len(mri_ds) - train_size
train_ds, val_ds = random_split(mri_ds, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# Initialize model
encoder = MRIEncoder(embed_dim=512)
classifier = nn.Linear(512, 5)  # 5 classes
model = nn.Sequential(encoder, classifier)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

best_val_loss = float('inf')
patience = 5
patience_counter = 0

# Training loop
epochs = 30
# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    for scan, _, label in train_loader:
        scan, label = scan.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(scan)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == label).sum().item()

    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Acc: {acc:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for scan, _, label in val_loader:
            scan, label = scan.to(device), label.to(device)
            out = model(scan)
            loss = criterion(out, label)
            val_loss += loss.item()
            val_correct += (out.argmax(1) == label).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(encoder.state_dict(), "checkpoints/pretrained_mri_encoder.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

