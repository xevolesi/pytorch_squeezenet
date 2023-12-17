import pathlib

import addict
import yaml


def read_config(config_path: pathlib.Path) -> addict.Dict:
    with config_path.open() as yaml_file:
        return addict.Dict(yaml.safe_load(yaml_file))
