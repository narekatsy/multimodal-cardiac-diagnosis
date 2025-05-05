import torch
import torch.nn as nn

class MetadataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embed_dim=512):
        super(MetadataEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3), # small dropout
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
