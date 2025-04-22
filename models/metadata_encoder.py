import torch
import torch.nn as nn

class MetadataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, embed_dim=64):
        super(MetadataEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
