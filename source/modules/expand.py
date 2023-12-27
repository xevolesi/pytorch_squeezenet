import torch
from torch import nn


class ExpandLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels_3x3: int, out_channels_1x1: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_3x3 = out_channels_3x3
        self.out_channels_1x1 = out_channels_1x1
        self.exp1x1 = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels_3x3, (1, 1)), nn.ReLU())
        self.exp3x3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels_3x3, (3, 3), padding=(1, 1)), nn.ReLU()
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.exp1x1(tensor), self.exp3x3(tensor)), dim=1)
