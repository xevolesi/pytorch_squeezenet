
import addict
import torch
from loguru import logger

from source.datasets.imagenet import create_dataloaders
from source.metrics import calculate_accuracy, calculate_top_5_accuracy
from source.utils.general import get_object_from_dict


def train(config: addict.Dict) -> None:
    dataloaders = create_dataloaders(config)
    if config.training.overfit_single_batch:
        single_batch = next(iter(dataloaders["train"]))
        dataloaders = {subset: [single_batch] for subset in dataloaders}

    device = torch.device(config.training.device)
    model: torch.nn.Module = get_object_from_dict(config.model).to(device)
    criterion: torch.nn.modules.loss._WeightedLoss = get_object_from_dict(config.criterion)
    optimizer: torch.optim.Optimizer = get_object_from_dict(config.optimizer, params=model.parameters())
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = get_object_from_dict(
        config.scheduler, optimizer=optimizer, total_iters=config.training.epochs
    )

    for epoch in range(config.training.epochs):
        training_loss = 0.0
        training_acc = 0.0
        training_acc_top5 = 0.0
        model.train()
        for batch in dataloaders["train"]:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            training_acc += calculate_accuracy(output, labels).item()
            training_acc_top5 += calculate_top_5_accuracy(output, labels).item()

        lr_scheduler.step()

        val_acc = 0.0
        validation_loss = 0.0
        val_acc_top5 = 0.0
        model.eval()
        for batch in dataloaders["val"]:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.no_grad():
                output = model(images)
                loss = criterion(output, labels)
                validation_loss += loss.item()
                val_acc += calculate_accuracy(output, labels).item()
                val_acc_top5 += calculate_top_5_accuracy(output, labels).item()

        logger.info(
            (
                "[EPOCH {cur_epoch}/{total_epochs}] TL: {tloss:.3f} VL: {vloss:.3f} TA: {tacc:.3f} VA: {vacc:.3f} "
                "TA5: {tacc5:.3f} VA5: {vacc5:.3f}"
            ),
            cur_epoch=epoch + 1,
            total_epochs=config.training.epochs,
            tloss=training_loss / len(dataloaders["train"]),
            vloss=validation_loss / len(dataloaders["val"]),
            tacc=training_acc / len(dataloaders["train"]),
            vacc=val_acc / len(dataloaders["val"]),
            tacc5=training_acc_top5 / len(dataloaders["train"]),
            vacc5=val_acc_top5 / len(dataloaders["val"]),
        )
