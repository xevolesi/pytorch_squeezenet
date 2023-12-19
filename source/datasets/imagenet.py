import json
from pathlib import Path

import addict
import albumentations as album
import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from source.utils.augmentations import get_albumentation_augs
from source.utils.custom_types import DatumDict

from .utils import DatasetMode


class ImageNet(Dataset):  # type: ignore[type-arg]
    def __init__(
        self,
        config: addict.Dict,
        mode: str = DatasetMode.TRAIN,
        transforms: album.Compose | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.config = config
        self.transforms = transforms
        self.root_dir = Path(config.path.dataset.root_dir) / f"{self.mode}"
        with Path(config.path.dataset.meta_file_path).open() as jfs:
            self.class_code_to_class_name: dict[str, str] = json.load(jfs)
        self.class_code_to_label = {code: idx for idx, code in enumerate(self.class_code_to_class_name.keys())}
        self.image_paths = [path.as_posix() for path in self.root_dir.iterdir()]
        self.labels = list(map(self.get_label_from_path, self.image_paths))

    def __getitem__(self, index: int) -> DatumDict:
        image_path = self.image_paths[index]

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_label = self.labels[index]

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        return {"image": image, "label": image_label}

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_class_code_from_path(self, path: str) -> str:
        return path.split("/")[-1].split("_")[-1].split(".")[0]

    def get_label_from_class_code(self, class_code: str) -> int:
        return self.class_code_to_label[class_code]

    def get_label_from_path(self, path: str) -> int:
        return self.get_label_from_class_code(self.get_class_code_from_path(path))


def create_dataloaders(config: addict.Dict) -> dict[str, DataLoader]:  # type: ignore[type-arg]
    dataloaders = {}
    augs = get_albumentation_augs(config)
    for subset in augs:
        dataset = ImageNet(config, mode=subset, transforms=augs[subset])
        dataloader = DataLoader(
            dataset,
            config.training.batch_size,
            shuffle=subset == DatasetMode.TRAIN,
            num_workers=config.training.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        dataloaders[subset] = dataloader
    return dataloaders
