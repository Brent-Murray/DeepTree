import os
import sys

import laspy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import MultiPoint

from .DGCNN import DGCNNFeatureExtractor
from .KPConv import KPConvExtractor
from .PointTransformer import PointTransformerFeatureExtractor
from .TabNet import TabNet
from .TreeExtractor import HybridParallelSequentialExtractor


# ------------------ Temperature Softmax ------------------
def temperature_softmax(logits, T=0.5):
    # Lower T -> sharper, Higher T -> flatter
    return F.softmax(logits / T, dim=-1)


# ------------------ Combined Feature Extractor ------------------
class CombinedFeatureExtractor(nn.Module):
    def __init__(
        self, model_dim, feature_dim, k_dgcnn=20, k_kpconv=16, k_pointtransformer=16
    ):
        super().__init__()
        self.dgcnn_feature_extractor = DGCNNFeatureExtractor(
            k=k_dgcnn, feature_dim=model_dim
        )
        self.kpconv_feature_extractor = KPConvExtractor(
            feature_dim=model_dim, k=k_kpconv
        )
        self.point_transformer_extractor = PointTransformerFeatureExtractor(
            k=k_pointtransformer, feature_dim=model_dim
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(model_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x):
        feat1 = self.dgcnn_feature_extractor(x)
        feat2 = self.kpconv_feature_extractor(x)
        feat3 = self.point_transformer_extractor(x)

        # Ensure features have the same shape, e.g. (B, feature_dim)
        if feat1.dim() == 3:
            feat1 = feat1.squeeze(-1)
        if feat2.dim() == 3:
            feat2 = feat2.squeeze(-1)
        if feat3.dim() == 3:
            feat3 = feat3.squeeze(-1)

        feats = torch.cat([feat1, feat2, feat3], dim=1)
        feats = self.fusion_mlp(feats)

        return feats


# ------------------ Multi-Head Tree Aggregator ------------------
class TreeEstimator(nn.Module):
    """
    A model with per-tree classification.
    """

    def __init__(
        self,
        num_species,
        input_dim,
        feature_dim=512,
        extractor="dgcnn",
    ):
        super().__init__()
        # 1) Per-tree feature extractor
        if extractor == "dgcnn":
            self.tree_feature_extractor = DGCNNFeatureExtractor(
                k=20, feature_dim=feature_dim, input_dim=8
            )
        elif extractor == "kpconv":
            self.tree_feature_extractor = KPConvExtractor(feature_dim=feature_dim, k=16)
        elif extractor == "pointtransformer":
            self.tree_feature_extractor = PointTransformerFeatureExtractor(
                feature_dim=feature_dim, k=16
            )
        elif extractor == "ensemble":
            self.tree_feature_extractor = CombinedFeatureExtractor(
                model_dim=128, feature_dim=feature_dim
            )
        elif extractor == "hybrid":
            self.tree_feature_extractor = HybridParallelSequentialExtractor(
                fusion_dim=256, pt_layers=4, pt_k=16
            )

        # 2) Per-tree classification head
        self.species_head = nn.Sequential(
            # nn.Linear(feature_dim + 320, 256),  # 80 for se4 320 for se5
            nn.Linear(128 + 320, 256),
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
        # self.tab_net = TabNet(input_dim, num_species)
        self.tab_net = TabNet(input_dim, 64, 64, 5, 1.5)

    def forward(self, x, metrics, return_masks=False):
        device = x.device
        B = x.size(0)
        x = x.permute(0, 2, 1).contiguous().to(device)
        x_feats = self.tree_feature_extractor(x)
        if return_masks:
            metrics_feats, masks = self.tab_net(metrics, return_masks)
        else:
            metrics_feats = self.tab_net(metrics)
        feats = torch.cat([x_feats, metrics_feats], dim=1)
        logits_tree = self.species_head(feats).contiguous()
        probs_tree = F.softmax(logits_tree, dim=-1).contiguous()
        species_argmax = torch.argmax(probs_tree, dim=-1)
        if return_masks:
            return (
                species_argmax,
                probs_tree,
                logits_tree,
                masks,
                x_feats,
                metrics_feats,
            )
        else:
            return species_argmax, probs_tree, logits_tree, x_feats, metrics_feats
