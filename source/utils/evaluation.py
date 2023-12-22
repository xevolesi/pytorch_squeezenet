from collections import defaultdict

import torch
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader

from source.metrics import calculate_accuracy, calculate_top_5_accuracy


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module, dataloader: DataLoader, criterion: _WeightedLoss, device: torch.device
) -> defaultdict[str, float]:
    model.eval()
    validation_result = defaultdict(lambda: torch.as_tensor(0.0, device=device))
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, labels)

        validation_result["VALIDATION_LOSS"] += loss.item()
        validation_result["VALIDATION_ACC@1"] += calculate_accuracy(output, labels)
        validation_result["VALIDATION_ACC@5"] += calculate_top_5_accuracy(output, labels)

    for result_name in validation_result:
        validation_result[result_name] /= len(dataloader)
        validation_result[result_name] = validation_result[result_name].item()

    return validation_result
