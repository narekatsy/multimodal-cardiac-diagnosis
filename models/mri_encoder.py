import torch
import torch.nn as nn
import torch.nn.functional as F

class MRIEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2):
        super(MRIEncoder, self).__init__()

        # 3D CNN to extract spatial features
        self.cnn3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),  # (B, 16, 128, 128, 16)
            nn.ReLU(),
            nn.MaxPool3d(2),  # (B, 16, 64, 64, 8)
            nn.Conv3d(16, 32, kernel_size=3, padding=1),  # (B, 32, 64, 64, 8)
            nn.ReLU(),
            nn.MaxPool3d(2),  # (B, 32, 32, 32, 4)
        )

        # Flatten to sequence
        self.flatten = nn.Flatten(2)  # from (B, C, D, H, W) → (B, C, D*H*W)
        self.linear_proj = nn.Linear(32, embed_dim)  # C → embed_dim

        # Positional encoding for sequence
        self.pos_embedding = nn.Parameter(torch.randn(1, 32 * 32 * 4, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 classes
        )

    def forward(self, x):
        # x shape: (B, 1, 128, 128, 16)
        x = self.cnn3d(x)  # (B, 32, 32, 32, 4)
        B, C, D, H, W = x.shape
        x = self.flatten(x)  # (B, C, D*H*W)
        x = x.permute(0, 2, 1)  # (B, Seq_len, C)
        x = self.linear_proj(x)  # (B, Seq_len, embed_dim)

        x += self.pos_embedding[:, :x.shape[1], :]

        x = self.transformer(x)  # (Seq_len, B, embed_dim)

        # Take mean over sequence for classification
        x = x.mean(dim=1)  # (B, embed_dim)
        out = self.cls_head(x)  # (B, num_classes)

        return out
