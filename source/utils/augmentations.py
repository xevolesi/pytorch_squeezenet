import addict
import albumentations as album
from albumentations.core.serialization import Serializable
from albumentations.pytorch.transforms import ToTensorV2

_DEFAULT_TRANSFORMS = album.Compose([album.Normalize(), ToTensorV2()])
_DEFAULT_TRANSFORM_SET = {subset: _DEFAULT_TRANSFORMS for subset in ("train", "val", "test")}


def get_albumentation_augs(config: addict.Dict) -> dict[str, Serializable]:
    """Build albumentations's augmentation pipelines from configuration file."""
    if not config.augmentations:
        return _DEFAULT_TRANSFORM_SET

    transforms = {}
    for subset in config.augmentations:
        transforms[subset] = album.from_dict(config.augmentations[subset])

    return transforms
