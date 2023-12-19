import pathlib

import addict
import pytest

from source.models import SqueezeNet
from source.utils.general import read_config

_CONFIG_PATH = "config.yml"


@pytest.fixture(scope="session")
def get_test_config() -> addict.Dict:
    config = read_config(pathlib.Path(_CONFIG_PATH))
    config.training.batch_size = 2
    return config


@pytest.fixture(scope="session")
def squeezenet(get_test_config: addict.Dict) -> SqueezeNet:
    return SqueezeNet(get_test_config.model.in_channels, get_test_config.model.num_classes)
