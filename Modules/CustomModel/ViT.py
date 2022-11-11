import torch
import torch.nn as nn
from torchvision import models
import timm

class ViT_Base(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.backbone.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        self.drop = nn.Dropout(0.4, inplace=True)

        # size
        self.size_fc = nn.Sequential(
            nn.Linear(2, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(128, 256)
        )

        # rgb 평균
        self.rgb_fc = nn.Sequential(
            nn.Linear(3, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(128, 256)
        )

        self.clf = nn.Sequential(
            nn.Linear(num_classes+256+256, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4, inplace=True),
            nn.Linear(1024, num_classes) # + len(장르, 국가)
        )
        
    def forward(self, x, size, rgb_mean):
        x = self.backbone(x)
        return x
