import pathlib

import addict
import torch

from source.utils.general import read_config, seed_everything
from source.utils.training import train

# Set benchmark to True and deterministic to False
# if you want to speed up training with less level of reproducibility.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Speed up GEMM if GPU allowed to use TF32.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(config: addict.Dict) -> None:
    seed_everything(config)
    train(config)


if __name__ == "__main__":
    config = read_config(pathlib.Path("config.yml"))
    main(config)
