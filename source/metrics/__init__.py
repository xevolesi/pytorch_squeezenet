import torch


@torch.no_grad()
def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    predicted_labels = logits.argmax(dim=1)
    return torch.eq(predicted_labels, labels).float().mean()


@torch.no_grad()
def calculate_top_5_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    batch_size = labels.size(0)
    _, y_pred = logits.topk(k=5, dim=1)
    y_pred = y_pred.t()
    target_reshaped = labels.view(1, -1).expand_as(y_pred)
    correct = y_pred == target_reshaped
    ind_which_topk_matched_truth = correct[:5]
    flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
    tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0)
    return tot_correct_topk / batch_size
