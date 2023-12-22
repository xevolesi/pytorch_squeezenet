import os
import pathlib

import addict
import torch

try:
    from dotenv import load_dotenv

    import wandb
except ImportError:
    load_dotenv = None
    wandb = None

from source.utils.general import read_config, seed_everything
from source.utils.training import train

# Set benchmark to True and deterministic to False
# if you want to speed up training with less level of reproducibility.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Speed up GEMM if GPU allowed to use TF32.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load envvars from .env.
if load_dotenv is not None and pathlib.Path(".env").exists():
    load_dotenv(".env")


def main(config: addict.Dict) -> None:
    if wandb is None or os.getenv("WANDB_API_KEY") is None:
        config.training.use_wandb = False
    run = wandb.init(project="SqueezeNet", config=config) if config.training.use_wandb else None
    seed_everything(config)
    train(config, run)
    run.finish()


if __name__ == "__main__":
    config = read_config(pathlib.Path("config.yml"))
    main(config)
