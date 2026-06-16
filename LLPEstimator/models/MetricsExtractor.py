import torch
import torch.nn as nn


class MetricsExtractor(nn.Module):
    def __init__(self, input_dim, n_d=64, n_a=64, n_steps=5, gamma=1.5, out_dim=128):
        """
        A simplified TabNet-like model.

        Args:
            input_dim (int): Number of input features.
            n_d (int): Dimension of the decision (feature representation) layer.
            n_a (int): Dimension of the attention layer.
            n_steps (int): Number of decision steps.
            gamma (float): Relaxation parameter for feature reusage.
            out_dim (int): Dimension of output features
        """
        super(MetricsExtractor, self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.bn = nn.LayerNorm(input_dim)
        # Initial transformation: splits into decision (n_d) and attention (n_a)
        self.initial_transform = nn.Linear(input_dim, n_d + n_a)

        # Feature transformer modules for each step
        self.feature_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.feature_transformers.append(
                nn.Sequential(
                    nn.Linear(n_d + n_a, 2 * (n_d + n_a)),
                    nn.LayerNorm(2 * (n_d + n_a)),
                    nn.GLU(),
                    nn.Linear(n_d + n_a, 2 * (n_d + n_a)),
                    nn.LayerNorm(2 * (n_d + n_a)),
                    nn.GLU(),
                )
            )

        # Attentive transformer modules: they take in attention features (n_a) and output a mask of size input_dim.
        self.attentive_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.attentive_transformers.append(
                nn.Sequential(
                    nn.Linear(n_a, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.Softmax(dim=1),
                )
            )

        # Final classifier on aggregated decision outputs from all steps
        self.proj = nn.Linear(n_d * n_steps, out_dim)

    def forward(self, x, return_masks=False):
        # x: (B, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.bn(x)
        # Initialize prior (all ones initially)
        prior = torch.ones(x.size(0), self.input_dim, device=x.device)
        # Initial transformation splits input into decision and attention features
        x_transformed = self.initial_transform(x)  # (B, n_d+n_a)
        decision = x_transformed[:, : self.n_d]  # (B, n_d)
        a = x_transformed[:, self.n_d :]  # (B, n_a)
        decisions = []
        masks = []

        for i in range(self.n_steps):
            # Apply attentive transformer to 'a' only, not a*prior
            mask = self.attentive_transformers[i](a)  # (B, input_dim)
            # Multiply mask element-wise with the prior
            mask = mask * prior  # (B, input_dim)
            # Update the prior: penalize features that have been used
            prior = prior * (self.gamma - mask)
            masks.append(mask)

            # Concatenate current decision and attention features and pass through feature transformer
            transformer_input = torch.cat([decision, a], dim=1)  # (B, n_d+n_a)
            transformer_output = self.feature_transformers[i](
                transformer_input
            )  # (B, n_d+n_a)
            decision = transformer_output[:, : self.n_d]  # (B, n_d)
            a = transformer_output[:, self.n_d :]  # (B, n_a)
            decisions.append(decision)

        # Aggregate decisions from all steps and classify
        aggregated = torch.cat(decisions, dim=1)  # (B, n_d * n_steps)
        out = self.proj(aggregated)
        if return_masks:
            return out, torch.stack(masks, dim=1)
        else:
            return out
