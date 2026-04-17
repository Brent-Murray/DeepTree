import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .PointExtractor import PointExtractor


# ------------------ Multi-Head Tree Aggregator ------------------
class TreeEstimatorPoint(nn.Module):
    """
    A model to estimate tree species from point cloud metrics
    """

    def __init__(self, num_species, first_dim, last_dim, layers, extractor="edgeconv"):
        super().__init__()

        # 1) PointExtractor
        self.tree_feature_extractor = PointExtractor(
            first_dim, last_dim, layers, extractor
        )
        # 2) Per-tree classification head
        self.species_head = nn.Sequential(
            nn.Linear(128, 256),
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

    def forward(self, x):
        device = x.device
        B = x.size(0)
        x = x.permute(0, 2, 1).contiguous().to(device)
        x_feats = self.tree_feature_extractor(x)

        logits_tree = self.species_head(x_feats).contiguous()
        probs_tree = F.softmax(logits_tree, dim=-1).contiguous()
        species_argmax = torch.argmax(probs_tree, dim=-1)

        return species_argmax, probs_tree, logits_tree, x_feats
