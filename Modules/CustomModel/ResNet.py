
import torch
import torch.nn as nn
from torchvision import models
# 224
class ResNet18_Artist_V0(nn.Module):
    def __init__(self, num_classes, is_freeze:bool=False):
        super().__init__()
        self.backbone = models.resnet18(wegits = models.ResNet18_Weights.IMAGENET1K_V1)
        if(is_freeze):
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1000, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(128, num_classes)
        )

        
    def forward(self, data):
        x = data['x']#, data['size'], data['rgb_mean']
        x = self.backbone(x)
        x = self.classifier(x)
        return x