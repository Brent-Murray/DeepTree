# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# class GLU(nn.Module):
#     def forward(self, x):
#         # Split input into two halves and apply GLU: x1 * sigmoid(x2)
#         dim = x.size(1) // 2
#         return x[:, :dim] * torch.sigmoid(x[:, dim:])


# class FourLayerFeatureTransformer(nn.Module):
#     def __init__(self, in_features):
#         super(FourLayerFeatureTransformer, self).__init__()
#         self.fc1 = nn.Linear(in_features, 2 * in_features, bias=False)
#         self.bn1 = nn.BatchNorm1d(2 * in_features)
#         self.fc2 = nn.Linear(in_features, 2 * in_features, bias=False)
#         self.bn2 = nn.BatchNorm1d(2 * in_features)
#         self.fc3 = nn.Linear(in_features, 2 * in_features, bias=False)
#         self.bn3 = nn.BatchNorm1d(2 * in_features)
#         self.fc4 = nn.Linear(in_features, 2 * in_features, bias=False)
#         self.bn4 = nn.BatchNorm1d(2 * in_features)
#         self.glu = GLU()
#         self.sqrt_half = math.sqrt(0.5)

#     def forward(self, x):
#         out1 = self.glu(self.bn1(self.fc1(x)))  # (B, in_features)
#         out2 = self.glu(self.bn2(self.fc2(out1)))  # (B, in_features)
#         out2 = (out2 + out1) * self.sqrt_half
#         out3 = self.glu(self.bn3(self.fc3(out2)))  # (B, in_features)
#         out3 = (out3 + out2) * self.sqrt_half
#         out4 = self.glu(self.bn4(self.fc4(out3)))  # (B, in_features)
#         out4 = (out4 + out3) * self.sqrt_half
#         return out4


# class TabNet(nn.Module):
#     def __init__(self, input_dim, n_d=64, n_a=64, n_steps=5, gamma=1.5):
#         """
#         A TabNet-like model with four-layer feature transformers.

#         Args:
#             input_dim (int): Number of input features.
#             output_dim (int): Decision output dimension (should match n_d).
#             n_d (int): Dimension of decision features.
#             n_a (int): Dimension of attention features.
#             n_steps (int): Number of decision steps.
#             gamma (float): Relaxation parameter for feature reusage.
#         """
#         super(TabNet, self).__init__()
#         self.input_dim = input_dim
#         self.n_d = n_d
#         self.n_a = n_a
#         self.n_steps = n_steps
#         self.gamma = gamma

#         self.bn = nn.BatchNorm1d(input_dim)
#         self.initial_transform = nn.Linear(input_dim, n_d + n_a)

#         # Create a four-layer transformer for each decision step.
#         self.feature_transformers = nn.ModuleList(
#             [FourLayerFeatureTransformer(n_d + n_a) for _ in range(n_steps)]
#         )

#         # Attentive transformers for mask computation (applied for steps < n_steps).
#         self.attentive_transformers = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Linear(n_a, input_dim, bias=False),
#                     nn.BatchNorm1d(input_dim),
#                     nn.Softmax(dim=1),
#                 )
#                 for _ in range(n_steps - 1)
#             ]
#         )

#     def forward(self, x, return_masks=False):
#         # x: (B, input_dim)
#         x = self.bn(x)
#         prior = torch.ones(x.size(0), self.input_dim, device=x.device)

#         # Initial transformation splits input into decision and attention features.
#         x_transformed = self.initial_transform(x)  # (B, n_d+n_a)
#         decision = x_transformed[:, : self.n_d]  # (B, n_d)
#         a = x_transformed[:, self.n_d :]  # (B, n_a)
#         decisions = []
#         masks = []

#         for i in range(self.n_steps):
#             # For steps before the last, compute and update the mask.
#             if i < self.n_steps - 1:
#                 mask = self.attentive_transformers[i](a)  # (B, input_dim)
#                 mask = mask * prior
#                 prior = prior * (self.gamma - mask)
#                 masks.append(mask)

#             # Concatenate decision and attention features and apply the four-layer transformer.
#             transformer_input = torch.cat([decision, a], dim=1)  # (B, n_d+n_a)
#             transformer_output = self.feature_transformers[i](transformer_input)
#             # Use ReLU on the decision part.
#             decision = F.relu(transformer_output[:, : self.n_d])
#             a = transformer_output[:, self.n_d :]
#             decisions.append(decision)

#         # Concatenate decisions from all steps.
#         aggregated = torch.cat(decisions, dim=1)
#         if return_masks:
#             return aggregated, torch.stack(masks, dim=1)
#         else:
#             return aggregated


import torch
import torch.nn as nn


class TabNet(nn.Module):
    def __init__(self, input_dim, n_d=64, n_a=64, n_steps=5, gamma=1.5):
        """
        A simplified TabNet-like model.

        Args:
            input_dim (int): Number of input features.
            n_d (int): Dimension of the decision (feature representation) layer.
            n_a (int): Dimension of the attention layer.
            n_steps (int): Number of decision steps.
            gamma (float): Relaxation parameter for feature reusage.
        """
        super(TabNet, self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.bn = nn.BatchNorm1d(input_dim)
        # Initial transformation: splits into decision (n_d) and attention (n_a)
        self.initial_transform = nn.Linear(input_dim, n_d + n_a)

        # Feature transformer modules for each step
        self.feature_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.feature_transformers.append(
                nn.Sequential(
                    nn.Linear(n_d + n_a, 2 * (n_d + n_a)),  # Double before GLU
                    nn.BatchNorm1d(2 * (n_d + n_a)),
                    nn.GLU(),  # Output: (batch_size, n_d + n_a)
                    nn.Linear(n_d + n_a, 2 * (n_d + n_a)),  # Double before GLU again
                    nn.BatchNorm1d(2 * (n_d + n_a)),
                    nn.GLU(),  # Output: (batch_size, n_d + n_a)
                )
            )

        # Attentive transformer modules: they take in attention features (n_a) and output a mask of size input_dim.
        self.attentive_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.attentive_transformers.append(
                nn.Sequential(
                    nn.Linear(n_a, input_dim),
                    nn.BatchNorm1d(input_dim),
                    nn.Softmax(dim=1),
                )
            )

        # # Final classifier on aggregated decision outputs from all steps
        # self.fc = nn.Linear(n_d * n_steps, output_dim)

    def forward(self, x, return_masks=False):
        # x: (B, input_dim)
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
        if return_masks:
            return aggregated, torch.stack(masks, dim=1)
        else:
            return aggregated
