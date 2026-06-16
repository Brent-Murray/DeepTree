import torch
import torch.nn as nn
import torch.nn.functional as F


class ROTLoss(nn.Module):
    def __init__(
        self,
        alpha=0.7,
        epsilon=1.0,
        sinkhorn_iter=75,
        eps=1e-8,
        label_smoothing=0.0,
        class_weights=None,
        temperature=1.0,
        ent_weight=1e-3,
    ):
        super().__init__()
        assert 0.0 < alpha < 1.0
        self.alpha = alpha
        self.epsilon = epsilon
        self.sinkhorn_iter = sinkhorn_iter
        self.eps = eps
        self.label_smoothing = float(label_smoothing)
        self.temperature = float(temperature)
        self.ent_weight = float(ent_weight)

        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w)
        else:
            self.register_buffer("class_weights", None)

    def forward(self, logits, target_props, mask=None):
        import torch.nn.functional as F

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            mask = (
                torch.ones(
                    logits.size(0),
                    logits.size(1),
                    dtype=torch.bool,
                    device=logits.device,
                )
                if mask is None
                else mask.unsqueeze(0)
            )

        B, n, K = logits.shape
        device = logits.device

        z = (
            target_props.unsqueeze(0).expand(B, -1).to(device)
            if target_props.dim() == 1
            else target_props.to(device)
        )
        if self.label_smoothing > 0:
            z = (1.0 - self.label_smoothing) * z + self.label_smoothing / K
            z = z / z.sum(dim=1, keepdim=True).clamp_min(self.eps)

        m = (
            torch.ones(B, n, dtype=torch.bool, device=device)
            if mask is None
            else mask.to(device).bool()
        )
        m_f = m.float()
        n_b = m_f.sum(dim=1, keepdim=True)  # [B,1]
        valid_bags = n_b.squeeze(1) > 0

        p = F.softmax(logits / self.temperature, dim=-1)  # [B,n,K]
        C = -torch.log(p + self.eps).permute(0, 2, 1)  # [B,K,n]
        K_mat = torch.exp(-C / self.epsilon) * m_f.unsqueeze(1)  # [B,K,n]

        tau = 1.0 / (1.0 + self.alpha * self.epsilon / (1.0 - self.alpha))
        b = m_f.clone()
        for _ in range(self.sinkhorn_iter):
            denom1 = torch.bmm(K_mat, b.unsqueeze(-1)).squeeze(-1) + self.eps  # [B,K]
            a = ((z * n_b) / denom1).clamp_min(self.eps).pow(tau)  # [B,K]
            denom2 = (
                torch.bmm(K_mat.transpose(1, 2), a.unsqueeze(-1)).squeeze(-1) + self.eps
            )  # [B,n]
            b = m_f / denom2

        U = a.unsqueeze(-1) * K_mat * b.unsqueeze(1)  # [B,K,n]
        cost_term = (C * U).sum(dim=(1, 2))
        H = -(U * (torch.log(U + self.eps) - 1.0)).sum(dim=(1, 2))
        inst_term = (
            self.alpha * (cost_term - self.epsilon * H) / n_b.squeeze(1).clamp_min(1.0)
        )

        prop_pred = U.sum(dim=2) / n_b
        prop_pred = prop_pred / prop_pred.sum(dim=1, keepdim=True).clamp_min(1e-8)

        if self.class_weights is None:
            w = torch.ones(K, device=device)
        else:
            w = self.class_weights.to(device)
        w = (w / w.mean().clamp_min(1e-12)).unsqueeze(0).expand_as(prop_pred)

        eps = self.eps
        targ = z.clamp_min(eps)
        pred = prop_pred.clamp_min(eps)
        kl_per_class = targ * (targ.log() - pred.log())
        bag_term = (1.0 - self.alpha) * (w * kl_per_class).sum(dim=1)

        ent = -(p.clamp_min(eps) * (p.clamp_min(eps)).log()).sum(dim=-1)
        ent = (ent * m_f).sum(dim=1) / n_b.squeeze(1).clamp_min(1.0)

        loss = inst_term + bag_term - self.ent_weight * ent
        return loss[valid_bags].mean() if valid_bags.any() else loss.new_tensor(0.0)


def plot_props_bias_aux_loss(
    plot_props, target_props, class_weights=None, beta=2.0, eps=1e-8
):
    """
    Weighted KL(target_props || plot_props) with bias-aware weights and optional class weights.
    Upweights:
      - high-proportion classes that are underpredicted
      - low-proportion classes that are overpredicted
    """
    targ = target_props.clamp_min(eps)
    pred = plot_props.clamp_min(eps)
    kl_per_class = targ * (targ.log() - pred.log())  # [B,K]

    # --- Bias-aware weighting term
    s = (targ - pred) * (2 * targ - 1.0)
    w_bias = torch.exp(beta * s)
    w_bias = w_bias / w_bias.mean(dim=1, keepdim=True).clamp_min(1e-12)

    # --- Class weights
    if class_weights is not None:
        w_class = torch.as_tensor(
            class_weights, dtype=torch.float32, device=plot_props.device
        )
        w_class = w_class / w_class.mean().clamp_min(1e-12)
        w_class = w_class.unsqueeze(0).expand_as(kl_per_class)  # [B,K]
    else:
        w_class = torch.ones_like(kl_per_class)

    # --- Combine weights and compute mean loss
    weighted_kl = w_bias * w_class * kl_per_class
    return weighted_kl.sum(dim=1).mean()  # scalar


class LLPLoss(nn.Module):
    """
    Wraps ROTLoss + aux term so you can create it ONCE and still call like calc_loss(...).

    Usage:
        loss_fn = LLPLoss(weights=class_weights, lambda_aux=0.1, beta=2.0, ...)
        total = loss_fn(logits, plot_props, y_true, mask)
    """

    def __init__(
        self,
        weights=None,
        lambda_aux: float = 0.1,
        beta: float = 2.0,
        # ROTLoss params (set defaults to your current values)
        rot_alpha: float = 0.3,
        rot_epsilon: float = 1.0,
        rot_sinkhorn_iter: int = 50,
        rot_eps: float = 1e-8,
        rot_label_smoothing: float = 0.05,
        rot_temperature: float = 1.0,
        rot_ent_weight: float = 1e-3,
    ):
        super().__init__()
        self.lambda_aux = float(lambda_aux)
        self.beta = float(beta)

        self.rot = ROTLoss(
            alpha=rot_alpha,
            epsilon=rot_epsilon,
            sinkhorn_iter=rot_sinkhorn_iter,
            eps=rot_eps,
            label_smoothing=rot_label_smoothing,
            class_weights=weights,
            temperature=rot_temperature,
            ent_weight=rot_ent_weight,
        )
        # Keep weights accessible for aux loss without recreating tensors each step
        self.register_buffer(
            "class_weights",
            (
                torch.as_tensor(weights, dtype=torch.float32)
                if weights is not None
                else None
            ),
            persistent=False,
        )

    def forward(self, logits, plot_props, y_true, mask):
        rot_val = self.rot(logits, y_true, mask)
        aux_val = plot_props_bias_aux_loss(
            plot_props, y_true, class_weights=self.class_weights, beta=self.beta
        )
        return rot_val + self.lambda_aux * aux_val


def calc_loss(logits, plot_props, y_true, mask, weights, lambda_aux=0.1, beta=2.0):
    """
    Combined loss: ROTLoss + bias-aware, class-weighted plot_props auxiliary loss.

    Args:
        logits: [B, T, K] per-tree logits (guided)
        plot_props: [B, K] plot-level predicted proportions from model
        y_true: [B, K] true plot-level species proportions
        mask: [B, T] tree mask (True for valid)
        weights: class weights (same tensor used for ROTLoss)
        lambda_aux: scalar weight for auxiliary term (0.05–0.3 typical)
        beta: strength of bias weighting in auxiliary term (1–3 typical)
    """
    rot_loss_fn = ROTLoss(
        alpha=0.3,
        epsilon=1.0,
        sinkhorn_iter=50,
        eps=1e-8,
        label_smoothing=0.05,
        class_weights=weights,
        temperature=1.0,
        ent_weight=1e-3,
    )

    rot_val = rot_loss_fn(logits, y_true, mask)
    aux_val = plot_props_bias_aux_loss(
        plot_props, y_true, class_weights=weights, beta=beta
    )

    total_loss = rot_val + lambda_aux * aux_val
    return total_loss
