import torch
from torch import nn

from .expand import ExpandLayer


class FireModule(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int, out_channels: int) -> None:
        """
        Args:
            in_channels: The number of input channels;
            squeeze_channels: The number of output channels for squeeze
            module;
            out_channels: The number of output channels for fire module.
            Must be divisible by 2 since it consists of the number of
            output channels of 1x1 expansion and 3x3 expansion.
        """
        super().__init__()
        if out_channels % 2 != 0:
            err_msg = f"`out_channels` must be divisible by 2, but got {out_channels}"
            raise ValueError(err_msg)
        self.squeeze = nn.Sequential(nn.Conv2d(in_channels, squeeze_channels, (1, 1)))
        self.relu = nn.ReLU(inplace=True)
        self.expand_layer = ExpandLayer(squeeze_channels, out_channels // 2, out_channels // 2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.expand_layer(self.relu(self.squeeze(tensor)))
