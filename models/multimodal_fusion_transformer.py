import torch
import torch.nn as nn

class MultimodalFusionTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, num_layers=2, num_classes=11):
        super(MultimodalFusionTransformer, self).__init__()

        self.embed_dim = embed_dim

        # Mask tokens for missing modalities
        self.mask_token_mri = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mask_token_ecg = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mask_token_meta = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, mri_emb=None, ecg_emb=None, meta_emb=None):
        B = mri_emb.shape[0] if mri_emb is not None else \
            ecg_emb.shape[0] if ecg_emb is not None else \
            meta_emb.shape[0]

        tokens = []

        if mri_emb is not None:
            tokens.append(mri_emb.unsqueeze(1))  # (B, 1, D)
        else:
            tokens.append(self.mask_token_mri.expand(B, 1, -1))

        if ecg_emb is not None:
            tokens.append(ecg_emb.unsqueeze(1))
        else:
            tokens.append(self.mask_token_ecg.expand(B, 1, -1))

        if meta_emb is not None:
            tokens.append(meta_emb.unsqueeze(1))
        else:
            tokens.append(self.mask_token_meta.expand(B, 1, -1))

        x = torch.cat(tokens, dim=1)  # (B, 3, D)
        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer(x)  # (B, 3, D)

        fused = x.mean(dim=1)  # (B, D)

        out = self.classifier(fused)  # (B, num_classes)
        return out
