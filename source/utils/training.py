from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    import addict
    from torch.nn.modules.loss import _WeightedLoss
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

try:
    from wandb.wandb_run import Run
except ImportError:
    Run = None

from source.datasets.imagenet import create_dataloaders
from source.metrics import calculate_accuracy, calculate_top_5_accuracy
from source.utils.evaluation import validate_one_epoch
from source.utils.general import get_cpu_state_dict, get_object_from_dict


def train(config: addict.Dict, wandb_run: Run | None = None) -> None:
    dataloaders = create_dataloaders(config)
    if config.training.overfit_single_batch:
        single_batch = next(iter(dataloaders["train"]))
        dataloaders = {subset: [single_batch] for subset in dataloaders}

    device = torch.device(config.training.device)
    model: torch.nn.Module = get_object_from_dict(config.model).to(device)
    criterion: _WeightedLoss = get_object_from_dict(config.criterion)
    optimizer: Optimizer = get_object_from_dict(config.optimizer, params=model.parameters())
    lr_scheduler: LRScheduler = get_object_from_dict(
        config.scheduler, optimizer=optimizer, total_iters=config.training.epochs
    )

    best_weights = None
    best_top1_acc = float("-inf")
    for epoch in range(config.training.epochs):
        training_result = train_one_epoch(model, dataloaders["train"], optimizer, criterion, device)
        lr_scheduler.step()
        validation_result = validate_one_epoch(model, dataloaders["val"], criterion, device)

        if validation_result["VALIDATION_ACC@1"] > best_top1_acc:
            best_top1_acc = validation_result["VALIDATION_ACC@1"]
            best_weights = get_cpu_state_dict(model)

        logger.info(
            (
                "[EPOCH {cur_epoch}/{total_epochs}] TL: {tloss:.3f} VL: {vloss:.3f} TA: {tacc:.3f} VA: {vacc:.3f} "
                "TA5: {tacc5:.3f} VA5: {vacc5:.3f}"
            ),
            cur_epoch=epoch + 1,
            total_epochs=config.training.epochs,
            tloss=training_result["TRAINING_LOSS"],
            tacc=training_result["TRAINING_ACC@1"],
            tacc5=training_result["TRAINING_ACC@5"],
            vacc=validation_result["VALIDATION_ACC@1"],
            vloss=validation_result["VALIDATION_LOSS"],
            vacc5=validation_result["VALIDATION_ACC@5"],
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "Training Loss": training_result["TRAINING_LOSS"],
                    "Training Acc@1": training_result["TRAINING_ACC@1"],
                    "Training Acc@5": training_result["TRAINING_ACC@5"],
                    "Validation Loss": validation_result["VALIDATION_LOSS"],
                    "Validation Acc@1": validation_result["VALIDATION_ACC@1"],
                    "Validation Acc@5": validation_result["VALIDATION_ACC@5"],
                    "LR": lr_scheduler.get_last_lr()[0],
                }
            )
    save_path = Path(config.path.weights_folder_path) / f"{wandb_run._run_id}"
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, (save_path / "state_dict.pth").as_posix())


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: _WeightedLoss,
    device: torch.device,
) -> defaultdict[str, float]:
    model.train()
    training_results = defaultdict(lambda: torch.as_tensor(0.0, device=device))
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        training_results["TRAINING_LOSS"] += loss
        training_results["TRAINING_ACC@1"] += calculate_accuracy(output, labels)
        training_results["TRAINING_ACC@5"] += calculate_top_5_accuracy(output, labels)

    for result_name in training_results:
        training_results[result_name] /= len(dataloader)
        training_results[result_name] = training_results[result_name].item()

    return training_results
