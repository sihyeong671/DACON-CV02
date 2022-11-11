import torch
import torch.nn as nn
from torchvision import models
import timm

class MaxViT_Base(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.maxvit_t(weights = models.MaxVit_T_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LayerNorm(512, eps=1e-05, elementwise_affine=True),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=num_classes, bias=False),
        )
        
    def forward(self, x, size, rgb_mean):
        x = self.backbone(x)

        return x

class MaxViT_SC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.maxvit_t(weights = models.MaxVit_T_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LayerNorm(512, eps=1e-05, elementwise_affine=True),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=num_classes, bias=False),
        )
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
        x = self.drop(x)

        lin = self.size_fc(size)
        rgb = self.rgb_fc(rgb_mean)

        x = torch.cat((x, lin, rgb), 1) # 여기 cat부분이 cnn부분 끝나고 분류하는 layer들어가기 전에 붙인 부분입니다.
        x = self.clf(x)
        return x


class MaxViT_S(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.maxvit_t(weights = models.MaxVit_T_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LayerNorm(512, eps=1e-05, elementwise_affine=True),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=num_classes, bias=False),
        )
        self.drop = nn.Dropout(0.4, inplace=True)

        # size
        self.size_fc = nn.Sequential(
            nn.Linear(2, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(128, 256)
        )

        self.clf = nn.Sequential(
            nn.Linear(num_classes+256, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4, inplace=True),
            nn.Linear(1024, num_classes) # + len(장르, 국가)
        )
        
    def forward(self, x, size, rgb_mean):
        x = self.backbone(x)
        x = self.drop(x)

        lin = self.size_fc(size)

        x = torch.cat((x, lin), 1) # 여기 cat부분이 cnn부분 끝나고 분류하는 layer들어가기 전에 붙인 부분입니다.
        x = self.clf(x)
        return x

class MaxViT_C(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.maxvit_t(weights = models.MaxVit_T_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LayerNorm(512, eps=1e-05, elementwise_affine=True),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=num_classes, bias=False),
        )
        self.drop = nn.Dropout(0.4, inplace=True)

        # rgb 평균
        self.rgb_fc = nn.Sequential(
            nn.Linear(3, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(128, 256)
        )

        self.clf = nn.Sequential(
            nn.Linear(num_classes+256, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4, inplace=True),
            nn.Linear(1024, num_classes) # + len(장르, 국가)
        )
        
    def forward(self, x, size, rgb_mean):
        x = self.backbone(x)
        x = self.drop(x)

        rgb = self.rgb_fc(rgb_mean)

        x = torch.cat((x, rgb), 1) # 여기 cat부분이 cnn부분 끝나고 분류하는 layer들어가기 전에 붙인 부분입니다.
        x = self.clf(x)
        return x