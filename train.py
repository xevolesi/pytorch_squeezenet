import pathlib

from source.utils.general import read_config
from source.utils.training import train

if __name__ == "__main__":
    config = read_config(pathlib.Path("config.yml"))
    train(config)
