import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecg_encoder import ECGEncoder
from datasets import ECGDataset
import torch.optim as optim

# Load Dataset
ecg_ds = ECGDataset(signal_dir='processed_data/ecg/training/signals',
                    meta_dir='processed_data/ecg/training/metadata',
                    label_dir='processed_data/ecg/training/labels')
ecg_loader = DataLoader(ecg_ds, batch_size=8, shuffle=True)

# Initialize model
encoder = ECGEncoder()
classifier = nn.Linear(512, 9)
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
    for signal, meta, label in ecg_loader:
        signal, label = signal.to(device), label.to(device)

        optimizer.zero_grad()
        out = model(signal)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == label).sum().item()

    acc = correct / len(ecg_loader.dataset)
    print(f"Epoch {epoch+1} | ECG Loss: {total_loss:.4f} | Acc: {acc:.4f}")

    torch.save(encoder.state_dict(), "checkpoints/pretrained_ecg_encoder.pt")

