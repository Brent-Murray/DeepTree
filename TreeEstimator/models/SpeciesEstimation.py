import os
import sys
import math
import laspy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import MultiPoint

from .PointExtractor import PointExtractor
from .TabNet import TabNet


# ------------------ Multi-Head Tree Aggregator ------------------
def get_layer_dims(first_dim, last_dim, num_layers):
    def round_to_multiple(n, multiple=4):
        return int(round(n / multiple)) * multiple

    # Try to enforce "double or same" rule if last_dim is a power-of-two multiple of first_dim.
    ratio = last_dim / first_dim
    if ratio.is_integer():
        ratio_int = int(ratio)
        # Check if ratio_int is a power of 2.
        if ratio_int != 0 and (ratio_int & (ratio_int - 1)) == 0:
            required_doublings = int(math.log2(ratio_int))
            # We need at least required_doublings transitions; there are (num_layers - 1) available.
            if required_doublings <= (num_layers - 1):
                # Distribute doubling events evenly over the transitions.
                # doubling_events will hold the layer indices (0-indexed) at which a doubling occurs.
                doubling_events = np.round(
                    np.linspace(0, num_layers - 1, required_doublings + 1)[1:]
                ).astype(int)
                dims = []
                for i in range(num_layers):
                    # Count how many doubling events have occurred up to (and including) index i.
                    doublings = np.sum(doubling_events <= i)
                    dims.append(first_dim * (2**doublings))
                # Ensure last layer is exactly last_dim.
                dims[-1] = last_dim
                dims = [round_to_multiple(d) for d in dims]
                return dims


class TreeEstimator(nn.Module):
    """
    A model for per-tree classification.
    """

    def __init__(
        self,
        num_species,
        first_dim,
        last_dim,
        layers,
        n_metrics,
        extractor="edgeconv",
        tabnet_hidden=64,
    ):
        super().__init__()

        # 1) Per-tree feature extractor
        self.tree_feature_extractor = PointExtractor(
            first_dim, last_dim, layers, extractor
        )

        # 2) Per-tree classification head
        tn_out_dim = tabnet_hidden * 5
        pe_out_dim = sum(get_layer_dims(first_dim, last_dim, layers))
        self.species_head = nn.Sequential(
            nn.Linear(pe_out_dim + tn_out_dim, 256),
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
        self.tab_net = TabNet(n_metrics, tabnet_hidden, tabnet_hidden, 5, 1.5)

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


class EnsembleTreeEstimator(nn.Module):
    def __init__(
        self,
        num_species,
        n_metrics,
        layers,
        n_ensemble,
        base_first_dim,
        base_last_dim,
        base_tabnet_hidden=64,
        extractor="dgcnn",
    ):
        super().__init__()
        self.ensemble_models = nn.ModuleList()
        dim_multiplyer = list(range(2, 2 + n_ensemble))
        layer_multiplyer = [1 * (2**i) for i in range(n_ensemble)]
        self.dim_x = []
        self.dim_m = []
        for i in range(n_ensemble):

            first_dim = int(base_first_dim * dim_multiplyer[i])
            last_dim = int(base_last_dim * dim_multiplyer[i])
            tabnet_hidden = int(base_tabnet_hidden * dim_multiplyer[i])
            n_layers = int(layers * layer_multiplyer[i])
            model = TreeEstimator(
                num_species=num_species,
                first_dim=first_dim,
                last_dim=last_dim,
                layers=n_layers,
                n_metrics=n_metrics,
                extractor=extractor,
                tabnet_hidden=tabnet_hidden,
            )
            self.ensemble_models.append(model)
            self.dim_x.append(last_dim)
            self.dim_m.append(tabnet_hidden * 5)

        self.x_fc = nn.Linear(sum(self.dim_x), 256)
        self.m_fc = nn.Linear(sum(self.dim_m), 256)
        self.final_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_species),
        )

    def forward(self, x, metrics, return_masks=False):
        ensemble_outputs = [
            model(x, metrics, return_masks=return_masks)
            for model in self.ensemble_models
        ]
        x_feats_list = [out[4] for out in ensemble_outputs]
        metrics_feats_list = [out[5] for out in ensemble_outputs]
        if return_masks:
            masks = [out[3] for out in ensemble_outputs]
            masks = torch.cat(masks, dim=1)

        x_feats = torch.cat(x_feats_list, dim=1)
        x_feats = self.x_fc(x_feats)
        metrics_feats = torch.cat(metrics_feats_list, dim=1)
        metrics_feats = self.m_fc(metrics_feats)
        agg_feats = torch.cat([x_feats, metrics_feats], dim=1)
        final_logits = self.final_fc(agg_feats)
        final_probs = F.softmax(final_logits, dim=-1)
        final_species_argmax = torch.argmax(final_probs, dim=-1)

        if return_masks:
            return (
                final_species_argmax,
                final_probs,
                final_logits,
                masks,
                x_feats,
                metrics_feats,
            )
        else:
            return (
                final_species_argmax,
                final_probs,
                final_logits,
                x_feats,
                metrics_feats,
            )
