import os
import random

import addict
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from source.datasets.imagenet import ImageNet, create_dataloaders
from source.datasets.utils import DatasetMode
from source.utils.augmentations import get_albumentation_augs
from source.utils.custom_types import DatumDict


def _assert_dataset_sample_without_augs(sample: DatumDict) -> None:
    assert "image" in sample
    assert "label" in sample
    assert isinstance(sample["image"], np.ndarray)
    assert isinstance(sample["label"], int)


def _assert_dataset_sample_with_augs(sample: DatumDict) -> None:
    assert "image" in sample
    assert "label" in sample
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["label"], int)
    assert sample["image"].shape == (3, 227, 227)
    assert sample["image"].dtype == torch.float32
    assert 0 <= sample["label"] <= 999


@pytest.mark.parametrize("mode", ["TRAIN", "VAL"])
def test_imagenet_without_augs(get_test_config: addict.Dict, mode: str) -> None:
    if os.getenv("IS_LOCAL_RUN") is None:
        pytest.skip("Skip data tests for CI")
    dataset = ImageNet(get_test_config, mode=DatasetMode[mode])
    assert len(dataset) == len(dataset.image_paths) == len(dataset.labels)
    random_samples = random.choices(dataset, k=10)
    for sample in random_samples:
        _assert_dataset_sample_without_augs(sample)


@pytest.mark.parametrize("mode", ["TRAIN", "VAL"])
def test_imagenet(get_test_config: addict.Dict, mode: str) -> None:
    if os.getenv("IS_LOCAL_RUN") is None:
        pytest.skip("Skip data tests for CI")
    augs = get_albumentation_augs(get_test_config)
    dataset = ImageNet(get_test_config, mode=DatasetMode[mode], transforms=augs[DatasetMode[mode]])
    assert len(dataset) == len(dataset.image_paths) == len(dataset.labels)
    random_samples = random.choices(dataset, k=10)
    for sample in random_samples:
        _assert_dataset_sample_with_augs(sample)


def _assert_dataloader(dataloader: DataLoader, batch_size: int) -> None:  # type: ignore[type-arg]
    for batch in dataloader:
        assert "image" in batch
        assert "label" in batch
        assert isinstance(batch["image"], torch.Tensor)
        assert isinstance(batch["label"], torch.Tensor)

        batch_dim_image = batch["image"].shape[0]
        batch_dim_label = batch["image"].shape[0]
        assert batch_dim_image == batch_dim_label
        assert batch_dim_image <= batch_size
        assert batch_dim_label <= batch_size
        assert batch["image"].shape[1:] == (3, 227, 227)
        assert batch["image"].dtype == torch.float32
        assert batch["label"].dtype == torch.long


def test_create_dataloaders(get_test_config: addict.Dict) -> None:
    if os.getenv("IS_LOCAL_RUN") is None:
        pytest.skip("Skip data tests for CI")
    get_test_config.training.batch_size = 32
    dataloaders = create_dataloaders(get_test_config)
    for loader in dataloaders.values():
        _assert_dataloader(loader, get_test_config.training.batch_size)
