import torch
from modules import FireModule
from torch import nn


class SqueezeNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=(7, 7), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2), ceil_mode=True),
            FireModule(96, 16, 128),
            FireModule(128, 16, 128),
            FireModule(128, 32, 256),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            FireModule(256, 32, 256),
            FireModule(256, 48, 384),
            FireModule(384, 48, 384),
            FireModule(384, 64, 512),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            FireModule(512, 64, 512),
            nn.Dropout(0.5),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, num_classes, (1, 1)),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._init_weights_as_in_paper()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.final_conv(self.features(tensor))), start_dim=1)

    def _init_weights_as_in_paper(self) -> None:
        """
        Initialize weights as in authors Caffe implementation.
        Authors implementation: https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/train_val.prototxt
        Caffe initializers: https://jihongju.github.io/2017/05/10/caffe-filler/#xavier-filler
        PyTorch initializers: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_

        Convolutions in fire module were initialized using `xavier` filler.
        Biases were initialized as zeroes.
        Final convolution was initialized using N(0.0, 0.01).
        """
        # It seems that we need to use torch.nn.init.kaiming_uniform_
        # to meet the formula: sqrt(3 / fan_mode). But it's not clear
        # what mode to use `fan_in` or `fan_out`.
        for module in self.features.modules():
            if not isinstance(module, nn.Conv2d):
                continue
            torch.nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            torch.nn.init.constant_(module.bias, 0)

        # Initialize final convolution with N(0.0, 0.01).
        torch.nn.init.normal_(self.final_conv[0].weight, 0.0, 0.01)
        torch.nn.init.constant_(self.final_conv[0].bias, 0.0)
