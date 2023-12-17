import addict
import pytest
import torch
from modules import ExpandLayer, FireModule

_RANDOM_TENSOR_SPATIAL_SIZE: tuple[int, int] = (27, 27)


def _assert_output(
    model_output: torch.Tensor, true_batch_size: int, true_output_channels: int, true_height: int, true_width: int
) -> None:
    batch_size, channels, height, width = model_output.shape
    assert batch_size == true_batch_size
    assert channels == true_output_channels
    assert height == true_height
    assert width == true_width


@pytest.mark.parametrize(("in_channels", "out_channels_1x1", "out_channels_3x3"), [(96, 64, 64), (128, 128, 128)])
def test_expand_layer(
    get_test_config: addict.Dict, in_channels: int, out_channels_1x1: int, out_channels_3x3: int
) -> None:
    expand_layer = ExpandLayer(in_channels, out_channels_3x3, out_channels_1x1)
    random_tensor = torch.randn((get_test_config.training.batch_size, in_channels, *_RANDOM_TENSOR_SPATIAL_SIZE))
    output = expand_layer(random_tensor)
    _assert_output(
        output, get_test_config.training.batch_size, out_channels_1x1 + out_channels_3x3, *_RANDOM_TENSOR_SPATIAL_SIZE
    )


@pytest.mark.parametrize(("in_channels", "squeeze_channels", "out_channels"), [(96, 16, 128), (128, 32, 256)])
def test_fire_module(get_test_config: addict.Dict, in_channels: int, squeeze_channels: int, out_channels: int) -> None:
    fire_module = FireModule(in_channels, squeeze_channels, out_channels)
    random_tensor = torch.randn((get_test_config.training.batch_size, in_channels, *_RANDOM_TENSOR_SPATIAL_SIZE))
    with torch.no_grad():
        output = fire_module(random_tensor)
    _assert_output(output, get_test_config.training.batch_size, out_channels, *_RANDOM_TENSOR_SPATIAL_SIZE)
