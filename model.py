import timm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class MelanomaResNet18(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = dropout
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = F.dropout(features, p=self.dropout, training=self.training)
        return self.classifier(features).squeeze(-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]


class MelanomaTimm(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.feature_dim = self.backbone.num_features
        self.dropout = dropout
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = F.dropout(features, p=self.dropout, training=self.training)
        return self.classifier(features).squeeze(-1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(backbone: str, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    if backbone == "resnet18":
        return MelanomaResNet18(pretrained=pretrained, dropout=dropout)
    return MelanomaTimm(backbone=backbone, pretrained=pretrained, dropout=dropout)
