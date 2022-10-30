
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


class EfficientnetB4V0(nn.Module):
    def __init__(self) -> None:
        super(EfficientnetB4V0, self).__init__()
        self.backborn = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(512, 50),
        )

    def forward(self, x):
        x = self.backborn(x)
        x = self.classifier(x)
        return x
