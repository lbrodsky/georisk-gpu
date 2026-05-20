import torch


def pixel_accuracy(preds, targets):

    preds = torch.argmax(preds, dim=1)

    correct = (preds == targets).float()

    return correct.mean().item()
