import torch
import torch.nn as nn
import torchvision.models.video as models_video

class MRIEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super(MRIEncoder, self).__init__()

        # Load pretrained 3D ResNet18
        self.backbone = models_video.r3d_18(pretrained=True)
        self.backbone.stem[0] = nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
        self.backbone.fc = nn.Identity()

        self.proj_head = nn.Linear(512, embed_dim)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        features = self.backbone(x)  # (B, 512)
        embeddings = self.proj_head(features)  # (B, embed_dim)
        return embeddings
