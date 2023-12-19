import pydoc
from copy import deepcopy

import addict
import albumentations as album
import pytest
from albumentations.core.serialization import Serializable
from albumentations.pytorch.transforms import ToTensorV2

from source.utils.augmentations import get_albumentation_augs

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _assert_default_transform_set(subset_transforms: album.Compose) -> None:
    assert len(subset_transforms) == 2
    for transform_idx, transform in enumerate(subset_transforms):
        if transform_idx == 0:
            assert isinstance(transform, album.Normalize)
            assert transform.mean == _IMAGENET_MEAN
            assert transform.std == _IMAGENET_STD
        elif transform_idx == 1:
            assert isinstance(transform, ToTensorV2)
        else:
            raise AssertionError


def _assert_augs_from_config(config: addict.Dict, subset_name: str, actual_augs: album.Compose) -> None:
    config_augs = config.augmentations[subset_name]
    assert config_augs["transform"]["__class_fullname__"] == "Compose"
    config_augs = config_augs["transform"]["transforms"]
    assert len(config_augs) == len(actual_augs)
    for config_aug, actual_aug in zip(config_augs, actual_augs, strict=True):
        # Hack with type(...) | Serializable specially for MyPy. :)
        assert isinstance(actual_aug, type(pydoc.locate(config_aug["__class_fullname__"])) | Serializable)


@pytest.mark.parametrize("default", [True, False])
def test_get_albumentation_augs(default: bool, get_test_config: addict.Dict) -> None:
    config = deepcopy(get_test_config)
    if default:
        config.augmentations = {}
    augs = get_albumentation_augs(config)

    if default:
        assert set(augs.keys()) == {"train", "val", "test"}
        for subset_name in augs:
            _assert_default_transform_set(augs[subset_name])
    else:
        assert set(augs.keys()) == set(config.augmentations.keys())
        for subset_name in augs:
            _assert_augs_from_config(config, subset_name, augs[subset_name])
