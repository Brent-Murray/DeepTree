import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, logits, target, hard_mining=False):
        # If target is one-hot encoded, convert to class indices.
        if target.dim() == logits.dim():
            target = target.argmax(dim=1)
        if self.weights is not None:
            weights = self.weights.to(logits.device)
        else:
            weights = None
        if hard_mining:
            loss = F.cross_entropy(logits, target, weight=weights, reduction="none")
        else:
            loss = F.cross_entropy(logits, target, weight=weights)
        return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels, hard_mining=False):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels)
        if hard_mining:
            loss = (features - centers_batch).pow(2).sum(dim=1) / 2.0
        else:
            loss = (features - centers_batch).pow(2).sum() / (2.0 * batch_size)
        return loss


def ce_loss(y_pred, y_true, weights=None, hard_mining=False):
    weighted_ce = WeightedCrossEntropyLoss(weights)
    loss = weighted_ce(y_pred, y_true, hard_mining)

    return loss


def center_loss(y_true, features, num_class, feat_dim, hard_mining=False):
    center_loss = CenterLoss(num_class, feat_dim, features.device)
    loss = center_loss(features, y_true, hard_mining)
    return loss * 0.0001
