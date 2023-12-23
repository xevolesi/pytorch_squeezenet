import enum

import cv2
import jpeg4py as jpeg
import numpy as np
from numpy.typing import NDArray


class DatasetMode(enum.StrEnum):
    # Should be exactly same with subsets in configuration file.
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def read_image(image_path: str) -> NDArray[np.uint8]:
    """
    Try to read image with `jpeg4py`. If it's not possible
    then read image with `OpenCV`.

    Returns: RGB image.
    """
    try:
        image = jpeg.JPEG(image_path).decode()
    except jpeg.JPEGRuntimeError:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image
