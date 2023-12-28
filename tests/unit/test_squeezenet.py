import addict
import pytest
import torch

from source.models import SqueezeNetV1, SqueezeNetV11


def _assert_output(model_output: torch.Tensor, true_batch_size: int, true_n_classes: int) -> None:
    batch_size, n_classes = model_output.shape
    assert batch_size == true_batch_size
    assert n_classes == true_n_classes


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_squeezenet(
    version: str, get_test_config: addict.Dict, squeezenetv1: SqueezeNetV1, squeezenetv11: SqueezeNetV11
) -> None:
    model = squeezenetv1
    if version == "1.1":
        model = squeezenetv11
    random_tensor = torch.randn(
        get_test_config.training.batch_size, 3, get_test_config.training.image_size, get_test_config.training.image_size
    )
    with torch.no_grad():
        output = model(random_tensor)
    _assert_output(output, get_test_config.training.batch_size, get_test_config.model.num_classes)
