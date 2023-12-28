import pathlib

import addict
import pytest

from source.models import SqueezeNetV1, SqueezeNetV11
from source.utils.general import read_config

_CONFIG_PATH = "config.yml"


@pytest.fixture(scope="session")
def get_test_config() -> addict.Dict:
    config = read_config(pathlib.Path(_CONFIG_PATH))
    config.training.batch_size = 2
    return config


@pytest.fixture(scope="session")
def squeezenetv1(get_test_config: addict.Dict) -> SqueezeNetV1:
    return SqueezeNetV1(get_test_config.model.in_channels, get_test_config.model.num_classes)


@pytest.fixture(scope="session")
def squeezenetv11(get_test_config: addict.Dict) -> SqueezeNetV11:
    return SqueezeNetV11(get_test_config.model.in_channels, get_test_config.model.num_classes)
