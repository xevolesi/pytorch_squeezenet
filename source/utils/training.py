from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytz
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
from source.models import ModelEmaV2
from source.utils.evaluation import validate_one_epoch
from source.utils.general import get_cpu_state_dict, get_object_from_dict


def train(config: addict.Dict, wandb_run: Run | None = None) -> None:  # noqa: PLR0915
    dataloaders = create_dataloaders(config)
    if config.training.overfit_single_batch:
        single_batch = next(iter(dataloaders["train"]))
        dataloaders = {subset: [single_batch] for subset in dataloaders}

    start_epoch = 0
    device = torch.device(config.training.device)
    model: torch.nn.Module = get_object_from_dict(config.model).to(device)
    criterion: _WeightedLoss = get_object_from_dict(config.criterion)
    optimizer: Optimizer = get_object_from_dict(config.optimizer, params=model.parameters())
    lr_scheduler: LRScheduler = get_object_from_dict(config.scheduler, optimizer=optimizer)

    model_ema = None
    if config.training.ema_coef is not None:
        model_ema = ModelEmaV2(model, decay=config.training.ema_coef, device=device)

    # Load necessary state dicts to resume training.
    if config.training.resume.checkpoint_path is not None:
        full_ckpt = torch.load(config.training.resume.checkpoint_path)
        model.load_state_dict(full_ckpt["model"])
        optimizer.load_state_dict(full_ckpt["optimizer"])
        criterion.load_state_dict(full_ckpt["criterion"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(full_ckpt["scheduler"])
        if model_ema is not None:
            model_ema.module.load_state_dict(full_ckpt["model_ema"])
            model_ema.decay = full_ckpt["model_ema_decay"]
        start_epoch = full_ckpt["epoch"] + 1
        logger.info("Resuming training from {epoch} epoch", epoch=start_epoch)

    best_weights = None
    best_top1_acc = float("-inf")

    # Decide artifact path.
    run_id = datetime.now(tz=pytz.utc).strftime("%m/%d/%Y-%H:%M:%S")
    if wandb_run is not None:
        run_id = wandb_run._run_id
    save_folder_path = Path(config.path.weights_folder_path) / f"{run_id}"
    save_folder_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, config.training.epochs):
        training_result = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion, device, lr_scheduler, model_ema
        )
        validation_result = validate_one_epoch(model, dataloaders["val"], criterion, device)

        # Evaluate EMA model.
        ema_validation_result = None
        if model_ema is not None:
            ema_validation_result = validate_one_epoch(model_ema.module, dataloaders["val"], criterion, device)
            ema_validation_result = {"EMA_" + name: value for name, value in ema_validation_result.items()}

        # Choose which model is better: EMA or normal for best model candidate.
        candidate_acc = validation_result["VALIDATION_ACC@1"]
        weights_to_save = get_cpu_state_dict(model)
        if (ema_validation_result is not None) and (ema_validation_result["EMA_VALIDATION_ACC@1"] >= candidate_acc):
            weights_to_save = get_cpu_state_dict(model_ema.module)
            candidate_acc = ema_validation_result["EMA_VALIDATION_ACC@1"]

        # Decide is current epoch result is better than previous best result?
        if candidate_acc > best_top1_acc:
            best_top1_acc = candidate_acc
            best_weights = weights_to_save

        # Logging stuff.
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
                    **ema_validation_result,
                },
                step=epoch,
            )

        # Save full checkpoint to be able to resume training.
        full_ckpt = {
            "epoch": epoch,
            "model": get_cpu_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "criterion": criterion.state_dict(),
            "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        }
        if model_ema is not None:
            full_ckpt |= {"model_ema": get_cpu_state_dict(model_ema.module), "model_ema_decay": model_ema.decay}
        training_checkpoint_folder_path = save_folder_path / "training_checkpoints"
        training_checkpoint_folder_path.mkdir(exist_ok=True, parents=True)
        torch.save(full_ckpt, (training_checkpoint_folder_path / f"full_ckpt_{epoch}.pth"))

    # Save best model only checkpoint.
    torch.save(best_weights, (save_folder_path / "state_dict.pth").as_posix())


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: _WeightedLoss,
    device: torch.device,
    lr_scheduler: LRScheduler | None = None,
    model_ema: ModelEmaV2 | None = None,
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

        if model_ema is not None:
            model_ema.update(model)

        training_results["TRAINING_LOSS"] += loss
        training_results["TRAINING_ACC@1"] += calculate_accuracy(output, labels)
        training_results["TRAINING_ACC@5"] += calculate_top_5_accuracy(output, labels)

        # As far as i understand in original paper learning ratw was
        # decreased linearly after each batch.
        if lr_scheduler is not None:
            lr_scheduler.step()

    for result_name in training_results:
        training_results[result_name] /= len(dataloader)
        training_results[result_name] = training_results[result_name].item()

    return training_results
