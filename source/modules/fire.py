import torch
from torch import Tensor, nn

from .expand import ExpandLayer


class FireModule(nn.Module):
    """
    Standard fire module from paper. Contains squeeze layer and expand
    layer with ReLU activations.
    """

    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        out_channels: int,
        bn_in_expand: bool = False,
        bn_in_squeeze: bool = False,
    ) -> None:
        """
        Args:
            in_channels: The number of input channels;
            squeeze_channels: The number of output channels for squeeze
            module;
            out_channels: The number of output channels for fire module.
            Must be divisible by 2 since it consists of the number of
            output channels of 1x1 expansion and 3x3 expansion;
            bn_in_expand: Whether to use BatchNormalization layer in
            expand layer;
            bn_in_squeeze: Whether to use BatchNormalization layer in
            squeeze layer.
        """
        super().__init__()
        if out_channels % 2 != 0:
            err_msg = f"`out_channels` must be divisible by 2, but got {out_channels}"
            raise ValueError(err_msg)

        squeeze_modules = [nn.Conv2d(in_channels, squeeze_channels, (1, 1)), nn.ReLU()]
        if bn_in_squeeze:
            squeeze_modules.append(nn.BatchNorm2d(squeeze_channels))
        self.squeeze = nn.Sequential(*squeeze_modules)

        self.expand_layer = ExpandLayer(squeeze_channels, out_channels // 2, out_channels // 2, bn_in_expand)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.expand_layer(self.squeeze(tensor))


class FirModuleSimpleSkip(FireModule):
    """
    Fire module with simple bypass connections from paper. Contains squeeze
    layer and expand layer with ReLU activations and simple elementwise
    addition as a bypass connection.
    """

    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        out_channels: int,
        bn_in_expand: bool = False,
        bn_in_squeeze: bool = False,
    ) -> None:
        """
        Args:
            in_channels: The number of input channels;
            squeeze_channels: The number of output channels for squeeze
            module;
            out_channels: The number of output channels for fire module.
            Must be divisible by 2 since it consists of the number of
            output channels of 1x1 expansion and 3x3 expansion;
            bn_in_expand: Whether to use BatchNormalization layer in
            expand layer;
            bn_in_squeeze: Whether to use BatchNormalization layer in
            squeeze layer.
        """
        super().__init__(in_channels, squeeze_channels, out_channels, bn_in_expand, bn_in_squeeze)
        if in_channels != out_channels:
            err_msg = (
                f"Argument `in_channels` must be equal to argument `out_channels`, "
                f"but got {in_channels} != {out_channels}"
            )
            raise ValueError(err_msg)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(out_channels, squeeze_channels, bias=False),
            nn.ReLU(),
            nn.Linear(squeeze_channels, out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        block_output = super().forward(tensor)
        b, c, *_ = block_output.size()
        se_weights = self.se(block_output).view(b, c, 1, 1)
        return block_output * se_weights + tensor


class FireModuleComplexSkip(FireModule):
    """
    Fire module with complex bypass connections from paper. Contains squeeze
    layer and expand layer with ReLU activations and bypass connection as a
    1x1 convolution with ReLU activation to match channels.
    """

    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        out_channels: int,
        bn_in_expand: bool = False,
        bn_in_squeeze: bool = False,
    ) -> None:
        """
        Args:
            in_channels: The number of input channels;
            squeeze_channels: The number of output channels for squeeze
            module;
            out_channels: The number of output channels for fire module.
            Must be divisible by 2 since it consists of the number of
            output channels of 1x1 expansion and 3x3 expansion;
            bn_in_expand: Whether to use BatchNormalization layer in
            expand layer;
            bn_in_squeeze: Whether to use BatchNormalization layer in
            squeeze layer.
        """
        super().__init__(in_channels, squeeze_channels, out_channels, bn_in_expand, bn_in_squeeze)

        # If main path has batch normalization layer then let the
        # complex skip connection has it either.
        skip_conv_modules = [nn.Conv2d(in_channels, out_channels, (1, 1)), nn.ReLU()]
        if bn_in_squeeze:
            skip_conv_modules.append(nn.BatchNorm2d(out_channels))
        self.skip_conv = nn.Sequential(*skip_conv_modules)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(out_channels, squeeze_channels, bias=False),
            nn.ReLU(),
            nn.Linear(squeeze_channels, out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        block_output = super().forward(tensor)
        b, c, *_ = block_output.size()
        se_weights = self.se(block_output).view(b, c, 1, 1)
        return block_output * se_weights + self.skip_conv(tensor)


def fire_module_factory(
    in_channels: int,
    squeeze_channels: int,
    out_channels: int,
    bn_in_expand: bool = False,
    bn_in_squeeze: bool = False,
    skip_connection_type: str | None = None,
) -> FireModule:
    if skip_connection_type not in {"simple", "complex", None}:
        err_msg = (
            f"Wrong value for `skip_connection_type` argument. Expected one of ('simple', 'complex', None), "
            f"but got {skip_connection_type}"
        )
        raise ValueError(err_msg)
    kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "squeeze_channels": squeeze_channels,
        "bn_in_expand": bn_in_expand,
        "bn_in_squeeze": bn_in_squeeze,
    }
    module = FireModule
    if skip_connection_type == "simple":
        module = FirModuleSimpleSkip
    elif skip_connection_type == "complex":
        module = FireModuleComplexSkip
    return module(**kwargs)
