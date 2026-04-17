import torch
import torch.nn as nn


# ------------------ K-Nearest Neighbours ------------------
def knn(x, k):
    # x: [B, C, N]
    B, C, N = x.size()
    # Permute to [B, N, C] for batched matmul
    x_perm = x.permute(0, 2, 1).contiguous()
    # Compute inner product using batched matrix multiplication
    inner = -2 * torch.bmm(x_perm, x)  # [B, N, N]
    xx = torch.sum(x**2, dim=1, keepdim=True)  # [B, 1, N]
    # Compute pairwise distances
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    # Get indices and force reallocation
    idx = pairwise_distance.topk(k=k, dim=-1)[1].clone().contiguous()
    return idx


# ------------------ Get Graph Features ------------------
def get_graph_feature(x, k=20, idx=None):
    # x: [B, C, N]
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.contiguous()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = (
        torch.arange(0, batch_size, device=device).view(-1, 1, 1).contiguous()
        * num_points
    )
    idx = (idx.contiguous() + idx_base.contiguous()).contiguous()
    idx = idx.view(-1).contiguous()

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # [B, N, C]
    feature = x.reshape(batch_size * num_points, -1)[idx, :].contiguous()  # [B*N*k, C]
    feature = feature.view(batch_size, num_points, k, num_dims).contiguous()
    x = (
        x.view(batch_size, num_points, 1, num_dims)
        .contiguous()
        .repeat(1, 1, k, 1)
        .contiguous()
    )
    combined = torch.cat((feature - x, x), dim=3).contiguous()  # [B, N, k, 2C]
    combined = combined.permute(0, 3, 1, 2).contiguous()  # [B, 2C, N, k]
    return combined


# ------------------ Normalize Point Clouds ------------------
def normalize_point_clouds(xyz):
    # xyz: [B, 3, N]
    B, C, N = xyz.shape
    centroids = xyz.mean(dim=2, keepdim=True).contiguous()  # [B, 3, 1]
    xyz_centered = (xyz - centroids).contiguous()
    max_distances = (
        (xyz_centered**2).sum(dim=1).sqrt().max(dim=1, keepdim=True)[0]
    ).contiguous()
    max_distances = max_distances.unsqueeze(-1).contiguous()  # [B, 1, 1]
    xyz_norm = (xyz_centered / (max_distances + 1e-8)).contiguous()
    return xyz_norm


# # ------------------ DGCNN Feature Extractor ------------------
# class Conv2dLayerNormReLU(nn.Module):
#     def __init__(
#         self, in_channels, out_channels, kernel_size=1, bias=False, negative_slope=0.2
#     ):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
#         self.ln = nn.LayerNorm(out_channels)  # normalized over channel dim only
#         self.act = nn.LeakyReLU(negative_slope)

#     def forward(self, x):
#         x = self.conv(x)  # x shape: (B, out_channels, H, W)
#         x = x.permute(0, 2, 3, 1)  # now (B, H, W, out_channels)
#         x = self.ln(x)
#         x = x.permute(0, 3, 1, 2)  # back to (B, out_channels, H, W)
#         return self.act(x)


# ------------------ DGCNN Feature Extractor ------------------
class Conv2dLayerNormReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, bias=False, negative_slope=0.2
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        self.ln = nn.LayerNorm(out_channels)  # normalized over channel dim only
        # self.ln = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)  # x shape: (B, out_channels, H, W)
        x = x.permute(0, 2, 3, 1)  # now (B, H, W, out_channels)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # back to (B, out_channels, H, W)
        return self.act(x)


# class DGCNNFeatureExtractor(nn.Module):
#     def __init__(self, k=20, feature_dim=256):
#         super().__init__()
#         self.k = k
#         self.conv1 = Conv2dLayerNormReLU(6, 16)
#         self.conv2 = Conv2dLayerNormReLU(16 * 2, 16)
#         self.conv3 = Conv2dLayerNormReLU(16 * 2, 32)
#         self.conv4 = Conv2dLayerNormReLU(32 * 2, 64)
#         self.align1 = nn.Conv1d(16, 32, 1, bias=False)
#         self.align2 = nn.Conv1d(32, 64, 1, bias=False)
#         self.fc = nn.Sequential(nn.Linear(256, feature_dim))

#     def forward(self, x):
#         """
#         Process each point cloud sample individually using only valid (unpadded) points.
#         Args:
#             x (Tensor): (B, 3, N) raw point coordinates.
#         Returns:
#             Tensor: (B, feature_dim) feature vector per sample.
#         """
#         outputs = []
#         B = x.size(0)
#         for i in range(B):
#             # Create mask for valid (nonzero) points.
#             valid_mask = x[i].abs().sum(dim=0) != 0
#             valid_x = x[i][:, valid_mask].unsqueeze(0)  # (1, 3, valid_N)
#             valid_x = normalize_point_clouds(valid_x)

#             # First layer: compute graph feature and conv1
#             gf1 = get_graph_feature(valid_x, k=self.k)
#             x1 = self.conv1(gf1).max(dim=-1)[0]

#             # Second layer
#             gf2 = get_graph_feature(x1, k=self.k)
#             x2 = self.conv2(gf2).max(dim=-1)[0]
#             x2 = x2 + x1

#             # Third layer
#             gf3 = get_graph_feature(x2, k=self.k)
#             x3 = self.conv3(gf3).max(dim=-1)[0]
#             x3 = x3 + self.align1(x2)

#             # Fourth layer
#             gf4 = get_graph_feature(x3, k=self.k)
#             x4 = self.conv4(gf4).max(dim=-1)[0]
#             x4 = x4 + self.align2(x3)

#             # Global pooling
#             x_cat = torch.cat([x1, x2, x3, x4], dim=1)
#             x_max = torch.max(x_cat, dim=-1)[0]
#             x_mean = torch.mean(x_cat, dim=-1)
#             x_pool = torch.cat([x_max, x_mean], dim=1)
#             out = self.fc(x_pool)
#             outputs.append(out)
#         return torch.cat(outputs, dim=0)  # (B, feature_dim)


class DGCNNFeatureExtractor(nn.Module):
    def __init__(self, k=20, feature_dim=256, input_dim=6):
        super().__init__()
        self.k = k
        self.conv1 = Conv2dLayerNormReLU(input_dim, 16)
        self.conv2 = Conv2dLayerNormReLU(16 * 2, 16)
        self.conv3 = Conv2dLayerNormReLU(16 * 2, 32)
        self.conv4 = Conv2dLayerNormReLU(32 * 2, 64)
        # self.align1 = nn.Conv1d(16, 32, 1, bias=False)
        # self.align2 = nn.Conv1d(32, 64, 1, bias=False)
        self.fc = nn.Sequential(nn.Linear(256, feature_dim))

    def forward(self, x):
        """
        Process each point cloud sample individually using only valid (unpadded) points.
        Args:
            x (Tensor): (B, 3, N) raw point coordinates.
        Returns:
            Tensor: (B, feature_dim) feature vector per sample.
        """
        outputs = []
        B = x.size(0)
        for i in range(B):
            # Create mask for valid (nonzero) points.
            valid_mask = x[i].abs().sum(dim=0) != 0
            valid_x = x[i][:, valid_mask].unsqueeze(0)  # (1, 3, valid_N)
            valid_x = normalize_point_clouds(valid_x)

            # First layer: compute graph feature and conv1
            gf1 = get_graph_feature(valid_x, k=self.k)
            x1 = self.conv1(gf1).max(dim=-1)[0]

            # Second layer
            gf2 = get_graph_feature(x1, k=self.k)
            x2 = self.conv2(gf2).max(dim=-1)[0]

            # Third layer
            gf3 = get_graph_feature(x2, k=self.k)
            x3 = self.conv3(gf3).max(dim=-1)[0]

            # Fourth layer
            gf4 = get_graph_feature(x3, k=self.k)
            x4 = self.conv4(gf4).max(dim=-1)[0]

            # Global pooling
            x_cat = torch.cat([x1, x2, x3, x4], dim=1)
            x_max = torch.max(x_cat, dim=-1)[0]
            # x_mean = torch.mean(x_cat, dim=-1)
            # x_pool = torch.cat([x_max, x_mean], dim=1)
            # out = self.fc(x_pool)
            outputs.append(x_max)
        return torch.cat(outputs, dim=0)  # (B, feature_dim)
