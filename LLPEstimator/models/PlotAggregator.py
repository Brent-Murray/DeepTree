import torch
import torch.nn as nn

class PlotAggregator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))  # global query
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, feats, mask):  # feats [B,T,F], mask [B,T]
        B, T, F = feats.shape
        q = self.query.expand(B, 1, F)  # [B,1,F]
        key_val = feats  # [B,T,F]
        key_padding = ~mask  # True = ignore
        pooled, _ = self.attn(
            q, key_val, key_val, key_padding_mask=key_padding
        )  # [B,1,F]
        pooled = pooled.squeeze(1)  # [B,F]
        return self.mlp(pooled)  # [B,F]