import typing as ty

import numpy as np
import torch
from numpy.typing import NDArray

ImageTensor: ty.TypeAlias = torch.Tensor
NumPyImage: ty.TypeAlias = NDArray[np.uint8]


class DatumDict(ty.TypedDict):
    image: ImageTensor | NumPyImage
    label: int
