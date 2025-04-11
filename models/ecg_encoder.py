import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2):
        super(ECGEncoder, self).__init__()

        # 1D Convolution to extract local features
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=15, out_channels=64, kernel_size=3, padding=1),  # (B, 64, input_length)
            nn.ReLU(),
            nn.MaxPool1d(2),  # Downsampling
            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # (B, 128, input_length / 2)
            nn.ReLU(),
            nn.MaxPool1d(2),  # Downsampling
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 9)  # 9 classes
        )

    def forward(self, x):
        # x shape: (B, 15, input_length)
        # Apply 1D convolution to extract local features
        x = self.conv1d(x)  # (B, 128, input_length / 4)

        # Flatten the output from conv layers (B, 128, input_length / 4) → (B, input_length / 4, 128)
        x = x.permute(0, 2, 1)  # (B, Seq_len, embed_dim)

        # Apply Transformer Encoder
        x = self.transformer(x)  # (B, Seq_len, embed_dim)

        # Take mean over sequence length for classification
        x = x.mean(dim=1)  # (B, embed_dim)

        # Apply classification head
        out = self.cls_head(x)  # (B, 5)
        return out
