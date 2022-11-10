import torch.nn as nn
import timm

class Regnet_Base(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('regnety_016', pretrained=True)

        self.clf = nn.Sequential(
            nn.Linear(1000, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x, size, rgb_mean):
        x = self.backbone(x)
        x = self.clf(x)
        return x

class Regnet_S(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('regnety_016', pretrained=True)

        self.clf = nn.Sequential(
            nn.Linear(1000, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x, size, rgb_mean):
        x = self.backbone(x)
        x = self.clf(x)
        return x


class Regnet_C(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('regnety_016', pretrained=True)

        self.clf = nn.Sequential(
            nn.Linear(1000, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x, size, rgb_mean):
        x = self.backbone(x)
        x = self.clf(x)
        return x


class Regnet_SC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('regnety_016', pretrained=True)

        self.clf = nn.Sequential(
            nn.Linear(1000, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x, size, rgb_mean):
        x = self.backbone(x)
        x = self.clf(x)
        return x