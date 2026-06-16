import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .MetricsExtractor import MetricsExtractor
from .PlotAggregator import PlotAggregator
from .PointExtractor import PointExtractor


def get_layer_dims(first_dim, last_dim, num_layers):
    def round_to_multiple(n, multiple=4):
        return int(round(n / multiple)) * multiple

    ratio = last_dim / first_dim
    if ratio.is_integer():
        ratio_int = int(ratio)
        if ratio_int != 0 and (ratio_int & (ratio_int - 1)) == 0:
            required_doublings = int(math.log2(ratio_int))
            if required_doublings <= (num_layers - 1):
                doubling_events = np.round(
                    np.linspace(0, num_layers - 1, required_doublings + 1)[1:]
                ).astype(int)
                dims = []
                for i in range(num_layers):
                    doublings = np.sum(doubling_events <= i)
                    dims.append(first_dim * (2**doublings))
                dims[-1] = last_dim
                dims = [round_to_multiple(d) for d in dims]
                return dims


class SpeciesLLP(nn.Module):
    def __init__(
        self,
        num_species,
        first_dim,
        last_dim,
        layers,
        n_metrics,
        mn_hidden,
        mn_out,
        feature_dim,
        guidance_strength: float = 0.5,
        eps: float = 1e-6,
        use_aggregator=False,
        plot_prop_mode="features",
        # blend between plot_from_feats and plot_from_trees
        plot_blend_alpha: float = 0.2,
        # used as *base* scale in balanced approach (kept for compatibility)
        plot_tree_temperature: float = 1.0,
        # NEW: minimum temperature when consensus is high (smaller => peakier)
        plot_tau_min: float = 0.7,
        # NEW: make blend alpha adapt based on consensus (recommended)
        adaptive_plot_blend: bool = True,
    ):
        super().__init__()
        assert plot_prop_mode in ("features", "logits"), "plot_prop_mode must be 'features' or 'logits'"
        self.num_species = num_species
        self.guidance_strength = float(guidance_strength)
        self.eps = float(eps)
        self.feature_dim = feature_dim
        self.use_aggregator = use_aggregator
        self.plot_prop_mode = plot_prop_mode

        self.plot_blend_alpha = float(plot_blend_alpha)
        self.plot_tree_temperature = float(plot_tree_temperature)

        # balanced approach params
        self.plot_tau_min = float(plot_tau_min)
        self.adaptive_plot_blend = bool(adaptive_plot_blend)

        self.point_extractor = PointExtractor(first_dim, last_dim, layers)

        self.metrics_extractor = MetricsExtractor(
            n_metrics, mn_hidden, mn_hidden, 5, 1.5, mn_out
        )

        mn_dim = mn_out
        pe_dim = sum(get_layer_dims(first_dim, last_dim, layers))
        fusion_in_dim = mn_dim + pe_dim
        fusion_hidden_dim = feature_dim * 2

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in_dim),
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, feature_dim),
        )

        self.species_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_species),
        )

        self.plot_aggregator = (
            PlotAggregator(feature_dim, feature_dim * 2, 4) if use_aggregator else None
        )

        self.plot_feat_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_species),
        )

    @torch.no_grad()
    def _make_masks(self, coords):
        point_mask = coords.abs().sum(dim=2) > 0  # [B,T,P]
        tree_mask = point_mask.any(dim=2)         # [B,T]
        return point_mask, tree_mask

    def forward(self, coords, metrics, return_preds=False, return_plot_props: bool = False):
        device = coords.device
        B, T, _, P = coords.shape

        if T == 0 or P == 0:
            logits = coords.new_zeros(B, T, self.num_species)
            tree_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            out = (logits, tree_mask)
            if return_plot_props:
                out = (*out, coords.new_zeros(B, self.num_species))
            if return_preds:
                pred_maps = [{} for _ in range(B)]
                out = (*out, pred_maps)
            return out

        point_mask, tree_mask = self._make_masks(coords)

        fused_logits = coords.new_zeros(B, T, self.num_species)   # [B,T,K]
        fused_feats  = coords.new_zeros(B, T, self.feature_dim)   # [B,T,F]

        # ---- Per-tree feature extraction + logits
        for b in range(B):
            valid_t = torch.nonzero(tree_mask[b], as_tuple=True)[0]
            for t in valid_t.tolist():
                pm = point_mask[b, t]          # [P]
                pts = coords[b, t, :, pm]      # [3, N_real_points]
                met = metrics[b, t]            # [M]

                pts_feat = self.point_extractor(pts)
                met_feat = self.metrics_extractor(met)

                if pts_feat.dim() > 1:
                    pts_feat = pts_feat.squeeze(0)
                if met_feat.dim() > 1:
                    met_feat = met_feat.squeeze(0)

                fused = torch.cat([pts_feat, met_feat], dim=-1)  # [pe_dim + mn_dim]
                fused = self.fusion(fused)                      # [feature_dim]

                fused_feats[b, t]  = fused
                fused_logits[b, t] = self.species_head(fused)    # [K]

        # ---- Plot-level proportions
        m = tree_mask.float().unsqueeze(-1)     # [B,T,1]
        n_b = m.sum(dim=1).clamp_min(1.0)       # [B,1,1]

        if self.plot_prop_mode == "logits":
            # SAME AS BEFORE
            plot_logits = (fused_logits * m).sum(dim=1) / n_b    # [B,K]
            plot_props = F.softmax(plot_logits, dim=-1)          # [B,K]
        else:
            # ===== BALANCED FEATURES-MODE =====
            # 1) tree probs for confidence + consensus estimates (no prob-averaging used in final plot_from_trees)
            p_tree = F.softmax(fused_logits, dim=-1)  # [B,T,K]

            # entropy-based confidence (high conf => more weight)
            ent = -(p_tree.clamp_min(self.eps) * p_tree.clamp_min(self.eps).log()).sum(dim=-1, keepdim=True)  # [B,T,1]
            ent = ent / math.log(self.num_species)  # [0,1]
            conf = (1.0 - ent)                      # [0,1]

            # consensus proxy: peak of the mean tree distribution
            p_mean = (p_tree * m).sum(dim=1) / n_b                 # [B,K]
            consensus = p_mean.max(dim=-1, keepdim=True).values    # [B,1] in [1/K, 1]

            # 2) confidence-weighted LOGIT aggregation (removes averaging probabilities)
            w = conf * m                                           # [B,T,1]
            w_sum = w.sum(dim=1).clamp_min(1.0)                    # [B,1]
            plot_logits_from_trees = (fused_logits * w).sum(dim=1) / w_sum  # [B,K]

            # 3) adaptive sharpening: only sharpen when consensus is high
            # map consensus from [1/K, 1] -> [0,1]
            c0 = 1.0 / self.num_species
            c = ((consensus - c0) / (1.0 - c0)).clamp(0.0, 1.0)    # [B,1]

            # tau_eff: 1.0 at low consensus, -> plot_tau_min at high consensus
            tau_min = max(self.plot_tau_min, 1e-3)
            tau_eff = 1.0 - c * (1.0 - tau_min)                    # [B,1]

            # keep compatibility with plot_tree_temperature as an overall scaling factor
            # (set plot_tree_temperature=1.0 if you don't want extra scaling)
            tau_eff = tau_eff * max(self.plot_tree_temperature, 1e-3)

            plot_from_trees = F.softmax(plot_logits_from_trees / tau_eff, dim=-1)  # [B,K]

            # 4) feature-head plot prediction (as before)
            if self.use_aggregator:
                plot_feats = self.plot_aggregator(fused_feats, tree_mask)  # [B,F]
            else:
                plot_feats = (fused_feats * m).sum(dim=1) / n_b            # [B,F]
            plot_logits_feats = self.plot_feat_proj(plot_feats)            # [B,K]
            plot_from_feats = F.softmax(plot_logits_feats, dim=-1)         # [B,K]

            # 5) blend (optionally adaptive)
            alpha0 = self.plot_blend_alpha
            if self.adaptive_plot_blend:
                # when trees disagree (low consensus), lean more on plot_from_feats
                alpha = (alpha0 + (1.0 - alpha0) * (1.0 - c)).clamp(0.0, 1.0)  # [B,1]
                plot_props = alpha * plot_from_feats + (1.0 - alpha) * plot_from_trees
            else:
                plot_props = alpha0 * plot_from_feats + (1.0 - alpha0) * plot_from_trees

            plot_props = plot_props / plot_props.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # ---- Guidance: DETACH prior + entropy-based confidence (applies in both modes)
        if self.guidance_strength != 0.0:
            prior = (plot_props.detach() + self.eps).log().unsqueeze(1)  # [B,1,K]

            with torch.no_grad():
                p = F.softmax(fused_logits, dim=-1)  # [B,T,K]
                ent = -(p.clamp_min(self.eps) * p.clamp_min(self.eps).log()).sum(
                    dim=-1, keepdim=True
                )  # [B,T,1]
                ent = ent / math.log(self.num_species)  # ~[0,1]

                conf_weight = (
                    ent * self.guidance_strength * tree_mask.unsqueeze(-1).float()
                )  # [B,T,1]

            logits_guided = fused_logits + conf_weight * prior
        else:
            logits_guided = fused_logits

        outputs = (logits_guided, tree_mask)
        if return_plot_props:
            outputs = (*outputs, plot_props)

        if return_preds:
            pred_maps: list[dict[int, int]] = []
            with torch.no_grad():
                for b in range(B):
                    bt = torch.nonzero(tree_mask[b], as_tuple=True)[0].tolist()
                    pred_b: dict[int, int] = {}
                    for t in bt:
                        cls_id = int(torch.argmax(logits_guided[b, t], dim=-1).item())
                        pred_b[t] = cls_id
                    pred_maps.append(pred_b)
            outputs = (*outputs, pred_maps)

        return outputs
