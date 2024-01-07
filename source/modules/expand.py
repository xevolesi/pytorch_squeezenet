import torch
from torch import nn


class ExpandLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels_3x3: int, out_channels_1x1: int, use_bn: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_3x3 = out_channels_3x3
        self.out_channels_1x1 = out_channels_1x1
        self.relu = nn.ReLU()
        self.exp1x1 = nn.Conv2d(self.in_channels, self.out_channels_1x1, (1, 1))
        self.exp3x3 = nn.Conv2d(self.in_channels, self.out_channels_3x3, (3, 3), padding=(1, 1))

        self.bn = None
        if use_bn:
            self.bn = nn.BatchNorm2d(self.out_channels_1x1 + self.out_channels_3x3)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.relu(torch.cat((self.exp1x1(tensor), self.exp3x3(tensor)), dim=1))
        if self.bn is not None:
            tensor = self.bn(tensor)
        return tensor
