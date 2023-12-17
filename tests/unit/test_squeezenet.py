import addict
import torch
from models import SqueezeNet


def _assert_output(model_output: torch.Tensor, true_batch_size: int, true_n_classes: int) -> None:
    batch_size, n_classes = model_output.shape
    assert batch_size == true_batch_size
    assert n_classes == true_n_classes


def test_squeezenet(get_test_config: addict.Dict) -> None:
    net = SqueezeNet(in_channels=3)
    random_tensor = torch.randn(
        get_test_config.training.batch_size, 3, get_test_config.training.image_size, get_test_config.training.image_size
    )
    with torch.no_grad():
        output = net(random_tensor)
    _assert_output(output, get_test_config.training.batch_size, get_test_config.model.num_classes)
