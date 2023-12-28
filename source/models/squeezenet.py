import torch
from torch import nn

from source.modules import FireModule

from .utils import do_paper_initialization


class SqueezeNetV1(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=(7, 7), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2), ceil_mode=True),
            FireModule(96, 16, 128),
            FireModule(128, 16, 128, apply_bypass_connections=True),
            FireModule(128, 32, 256),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            FireModule(256, 32, 256, apply_bypass_connections=True),
            FireModule(256, 48, 384),
            FireModule(384, 48, 384, apply_bypass_connections=True),
            FireModule(384, 64, 512),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            FireModule(512, 64, 512, apply_bypass_connections=True),
            nn.Dropout(0.5),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, num_classes, (1, 1)),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        do_paper_initialization(self)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.final_conv(self.features(tensor))), start_dim=1)


class SqueezeNetV11(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2), ceil_mode=True),
            FireModule(64, 16, 128),
            FireModule(128, 16, 128, apply_bypass_connections=False),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            FireModule(128, 32, 256),
            FireModule(256, 32, 256, apply_bypass_connections=False),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            FireModule(256, 48, 384),
            FireModule(384, 48, 384, apply_bypass_connections=False),
            FireModule(384, 64, 512),
            FireModule(512, 64, 512, apply_bypass_connections=False),
            nn.Dropout(0.5),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, num_classes, (1, 1)),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        do_paper_initialization(self)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.final_conv(self.features(tensor))), start_dim=1)
