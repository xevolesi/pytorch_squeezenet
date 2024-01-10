from torch import nn


class SEBlock(nn.Sequential):
    def __init__(self, in_channels: int, squeeze_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(in_channels, squeeze_channels, bias=False),
            nn.ReLU(),
            nn.Linear(squeeze_channels, in_channels, bias=False),
            nn.Sigmoid(),
        )
