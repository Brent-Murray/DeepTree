import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .TabNet import TabNet


# ------------------ Multi-Head Tree Aggregator ------------------
class TreeEstimatorMetrics(nn.Module):
    """
    A model to estimate tree species from point cloud metrics
    """

    def __init__(self, num_species, input_dim):
        super().__init__()
        # 1) Tabnet to process point cloud metrics
        self.tab_net = TabNet(input_dim, 64, 64, 5, 1.5)

        # 2) Per-tree classification head
        self.species_head = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_species),
        )

    def forward(self, metrics, return_masks=False):
        device = metrics.device
        B = metrics.size(0)

        if return_masks:
            metrics_feats, masks = self.tab_net(metrics, return_masks)
        else:
            metrics_feats = self.tab_net(metrics)

        logits_tree = self.species_head(metrics_feats).contiguous()
        probs_tree = F.softmax(logits_tree, dim=-1).contiguous()
        species_argmax = torch.argmax(probs_tree, dim=-1)
        if return_masks:
            return (
                species_argmax,
                probs_tree,
                logits_tree,
                masks,
                metrics_feats,
            )
        else:
            return species_argmax, probs_tree, logits_tree, metrics_feats
