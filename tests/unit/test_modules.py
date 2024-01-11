from itertools import product

import addict
import pytest
import torch

from source.modules import ExpandLayer, fire_module_factory
from source.modules.fire import FireModule, FireModuleComplexSkip, FirModuleSimpleSkip

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


@pytest.mark.parametrize(
    ("in_channels", "squeeze_channels", "out_channels", "use_se_block", "bn_in_expand", "bn_in_squeeze", "skip_type"),
    product(
        (96, 128, 256),
        (16, 32),
        (128, 256),
        (True, False),
        (True, False),
        (True, False),
        ("simple", "complex", None, "none"),
    ),
)
def test_fire_module_factory(
    in_channels: int,
    squeeze_channels: int,
    out_channels: int,
    use_se_block: bool,
    bn_in_expand: bool,
    bn_in_squeeze: bool,
    skip_type: str | None,
) -> None:
    if skip_type is None:
        module = fire_module_factory(
            in_channels, squeeze_channels, out_channels, use_se_block, bn_in_expand, bn_in_squeeze, skip_type
        )
        assert isinstance(module, FireModule)
    elif skip_type == "simple":
        if in_channels != out_channels:
            with pytest.raises(ValueError, match=r".+?"):
                module = fire_module_factory(
                    in_channels, squeeze_channels, out_channels, use_se_block, bn_in_expand, bn_in_squeeze, skip_type
                )
        else:
            module = fire_module_factory(
                in_channels, squeeze_channels, out_channels, use_se_block, bn_in_expand, bn_in_squeeze, skip_type
            )
            assert isinstance(module, FirModuleSimpleSkip)
    elif skip_type == "complex":
        module = fire_module_factory(
            in_channels, squeeze_channels, out_channels, use_se_block, bn_in_expand, bn_in_squeeze, skip_type
        )
        assert isinstance(module, FireModuleComplexSkip)
    else:
        with pytest.raises(ValueError, match=r".+?"):
            module = fire_module_factory(
                in_channels, squeeze_channels, out_channels, use_se_block, bn_in_expand, bn_in_squeeze, skip_type
            )


@pytest.mark.parametrize(
    ("in_channels", "squeeze_channels", "out_channels", "use_se_block", "bn_in_expand", "bn_in_squeeze"),
    product((96, 128), (16, 32), (128, 256), (True, False), (True, False), (True, False)),
)
def test_fire_module(
    get_test_config: addict.Dict,
    in_channels: int,
    squeeze_channels: int,
    out_channels: int,
    use_se_block: bool,
    bn_in_expand: bool,
    bn_in_squeeze: bool,
) -> None:
    fire_module = fire_module_factory(
        in_channels, squeeze_channels, out_channels, use_se_block, bn_in_expand, bn_in_squeeze
    )
    random_tensor = torch.randn((get_test_config.training.batch_size, in_channels, *_RANDOM_TENSOR_SPATIAL_SIZE))
    with torch.no_grad():
        output = fire_module(random_tensor)
    _assert_output(output, get_test_config.training.batch_size, out_channels, *_RANDOM_TENSOR_SPATIAL_SIZE)


@pytest.mark.parametrize(
    ("in_channels", "squeeze_channels", "out_channels", "use_se_block", "bn_in_expand", "bn_in_squeeze"),
    product((128,), (16, 32), (128,), (True, False), (True, False), (True, False)),
)
def test_fire_module_with_simple_skips(
    get_test_config: addict.Dict,
    in_channels: int,
    squeeze_channels: int,
    out_channels: int,
    use_se_block: bool,
    bn_in_expand: bool,
    bn_in_squeeze: bool,
) -> None:
    fire_module = fire_module_factory(
        in_channels,
        squeeze_channels,
        out_channels,
        use_se_block,
        bn_in_expand,
        bn_in_squeeze,
        skip_connection_type="simple",
    )
    random_tensor = torch.randn((get_test_config.training.batch_size, in_channels, *_RANDOM_TENSOR_SPATIAL_SIZE))
    with torch.no_grad():
        output = fire_module(random_tensor)
    _assert_output(output, get_test_config.training.batch_size, out_channels, *_RANDOM_TENSOR_SPATIAL_SIZE)


@pytest.mark.parametrize(
    ("in_channels", "squeeze_channels", "out_channels", "use_se_block", "bn_in_expand", "bn_in_squeeze"),
    product((96, 128), (16, 32), (128, 256), (True, False), (True, False), (True, False)),
)
def test_fire_module_with_complex_skips(
    get_test_config: addict.Dict,
    in_channels: int,
    squeeze_channels: int,
    out_channels: int,
    use_se_block: bool,
    bn_in_expand: bool,
    bn_in_squeeze: bool,
) -> None:
    fire_module = fire_module_factory(
        in_channels,
        squeeze_channels,
        out_channels,
        use_se_block,
        bn_in_expand,
        bn_in_squeeze,
        skip_connection_type="complex",
    )
    random_tensor = torch.randn((get_test_config.training.batch_size, in_channels, *_RANDOM_TENSOR_SPATIAL_SIZE))
    with torch.no_grad():
        output = fire_module(random_tensor)
    _assert_output(output, get_test_config.training.batch_size, out_channels, *_RANDOM_TENSOR_SPATIAL_SIZE)
