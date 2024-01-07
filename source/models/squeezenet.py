import torch
from torch import nn

from source.modules import fire_module_factory

from .utils import do_paper_initialization


class SqueezeNetV1(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, skip_connection_type: str | None = None) -> None:
        super().__init__()

        simple_skips = None
        complex_skips = None
        if skip_connection_type == "simple":
            simple_skips = "simple"
            complex_skips = None
        elif skip_connection_type == "complex":
            simple_skips = "simple"
            complex_skips = "complex"

        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=(7, 7), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2), ceil_mode=True),
            fire_module_factory(96, 16, 128, complex_skips),
            fire_module_factory(128, 16, 128, simple_skips),
            fire_module_factory(128, 32, 256, complex_skips),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            fire_module_factory(256, 32, 256, simple_skips),
            fire_module_factory(256, 48, 384, complex_skips),
            fire_module_factory(384, 48, 384, simple_skips),
            fire_module_factory(384, 64, 512, complex_skips),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            fire_module_factory(512, 64, 512, simple_skips),
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
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, skip_connection_type: str | None = None) -> None:
        super().__init__()

        simple_skips = None
        complex_skips = None
        if skip_connection_type == "simple":
            simple_skips = "simple"
            complex_skips = None
        elif skip_connection_type == "complex":
            simple_skips = "simple"
            complex_skips = "complex"

        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2), ceil_mode=True),
            fire_module_factory(64, 16, 128, complex_skips),
            fire_module_factory(128, 16, 128, simple_skips),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            fire_module_factory(128, 32, 256, complex_skips),
            fire_module_factory(256, 32, 256, simple_skips),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            fire_module_factory(256, 48, 384, complex_skips),
            fire_module_factory(384, 48, 384, simple_skips),
            fire_module_factory(384, 64, 512, complex_skips),
            fire_module_factory(512, 64, 512, simple_skips),
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


class SqueezeNetV12(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, skip_connection_type: str | None = None) -> None:
        super().__init__()

        simple_skips = None
        complex_skips = None
        if skip_connection_type == "simple":
            simple_skips = "simple"
            complex_skips = None
        elif skip_connection_type == "complex":
            simple_skips = "simple"
            complex_skips = "complex"

        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3, 3), stride=(2, 2)),
            nn.MaxPool2d((3, 3), stride=(2, 2), ceil_mode=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            fire_module_factory(64, 16, 128, True, True, complex_skips),
            fire_module_factory(128, 16, 128, True, True, simple_skips),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            fire_module_factory(128, 32, 256, True, True, complex_skips),
            fire_module_factory(256, 32, 256, True, True, simple_skips),
            nn.MaxPool2d((3, 3), stride=2, ceil_mode=True),
            fire_module_factory(256, 48, 384, True, True, complex_skips),
            fire_module_factory(384, 48, 384, True, True, simple_skips),
            fire_module_factory(384, 64, 512, True, True, complex_skips),
            fire_module_factory(512, 64, 512, True, True, simple_skips),
            nn.Dropout(0.5),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.avgpool(self.features(tensor)), 1))
