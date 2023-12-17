import pathlib

import addict
import pytest
from utils.general import read_config

_CONFIG_PATH = "config.yml"


@pytest.fixture(scope="module")
def get_test_config() -> addict.Dict:
    config = read_config(pathlib.Path(_CONFIG_PATH))
    config.training.batch_size = 2
    return config
