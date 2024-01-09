import typing as ty
from copy import deepcopy

import torch
from torch import nn


class ModelEmaV2(nn.Module):
    """Borrowed from here: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py"""

    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None) -> None:
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    @torch.no_grad()
    def _update(self, model: nn.Module, update_fn: ty.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(update_fn(ema_v, model_v.to(device=self.device)))

    def update(self, model: nn.Module) -> None:
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)
