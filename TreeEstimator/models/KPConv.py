import torch
import torch.nn as nn


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


def knn_query(coords, k, exclude_self=True):
    """
    Computes k‑nearest neighbors using only base PyTorch.

    Args:
        coords (Tensor): (B, N, 3) point coordinates.
        k (int): Number of neighbors.
        exclude_self (bool): Exclude the query point from its neighbor list.

    Returns:
        Tensor: Indices of neighbors, shape (B, N, k).
    """
    B, N, _ = coords.shape
    dists = torch.cdist(coords, coords)  # (B, N, N)

    if exclude_self:
        mask = torch.eye(N, device=coords.device).bool().unsqueeze(0).expand(B, N, N)
        dists = dists.masked_fill(mask, float("inf"))

    _, knn_idx = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
    return knn_idx


class KPConvEfficient(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_kernel_points=15,
        sigma=1.0,
        k=16,
        bias=True,
    ):
        """
        KPConv layer that uses only coordinates (both for neighbor queries and features).

        Args:
            in_channels (int): Number of input feature channels.
                               (For the first layer, use 3 since features = coords.)
            out_channels (int): Number of output feature channels.
            num_kernel_points (int): Number of kernel points.
            sigma (float): Gaussian sigma for influence weights.
            k (int): Number of neighbors to query.
            bias (bool): Whether to include a bias term.
        """
        super(KPConvEfficient, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernel_points = num_kernel_points
        self.sigma = sigma
        self.k = k

        # Fixed kernel points sampled randomly on the unit sphere.
        kernel_points = torch.randn(num_kernel_points, 3)
        kernel_points = kernel_points / kernel_points.norm(dim=1, keepdim=True)
        self.register_parameter(
            "kernel_points", nn.Parameter(kernel_points, requires_grad=False)
        )

        # Learnable kernel weights: (K, in_channels, out_channels)
        self.kernel_weights = nn.Parameter(
            torch.randn(num_kernel_points, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, coords, features):
        """
        Args:
            coords (Tensor): (B, 3, N) raw point coordinates.
            features (Tensor): (B, in_channels, N) features (for the first layer, these are the coords).
        Returns:
            Tensor: (B, out_channels, N) output features.
        """

        B, _, N = coords.shape
        # Rearranging to (B, N, 3) and (B, N, in_channels)
        X = coords.transpose(1, 2).contiguous()  # (B, N, 3)
        feat = features.transpose(1, 2).contiguous()  # (B, N, in_channels)

        # Get k-NN indices (B, N, k)
        nbr_idx = knn_query(X, self.k, exclude_self=True)

        # Use advanced indexing to gather neighbor coordinates.
        batch_indices = (
            torch.arange(B, device=X.device).view(B, 1, 1).expand(B, N, self.k)
        )
        neighbor_coords = X[batch_indices, nbr_idx, :]  # (B, N, k, 3)
        query = X.unsqueeze(2)  # (B, N, 1, 3)
        diff = neighbor_coords - query  # (B, N, k, 3)

        # Compute Gaussian influence weights relative to fixed kernel points.
        diff_exp = diff.unsqueeze(3)  # (B, N, k, 1, 3)
        kp = self.kernel_points.view(
            1, 1, 1, self.num_kernel_points, 3
        )  # (1, 1, 1, K, 3)
        diff_rel = diff_exp - kp  # (B, N, k, K, 3)
        sq_dist = (diff_rel**2).sum(dim=-1)  # (B, N, k, K)
        influence = torch.exp(-sq_dist / (self.sigma**2))  # (B, N, k, K)

        # Gather neighbor features using advanced indexing.
        neighbor_features = feat[batch_indices, nbr_idx, :]  # (B, N, k, in_channels)

        # Aggregate features per kernel point.
        aggregated = torch.einsum(
            "bnki,bnkc->bnic", influence, neighbor_features
        )  # (B, N, K, in_channels)
        out = torch.einsum(
            "bnki,kio->bno", aggregated, self.kernel_weights
        )  # (B, N, out_channels)
        if self.bias is not None:
            out = out + self.bias
        return out.transpose(1, 2)  # (B, out_channels, N)


class KPConvExtractor(nn.Module):
    def __init__(self, feature_dim=256, k=16):
        """
        A KPConv classification network that uses only raw coordinates.
        """
        super(KPConvExtractor, self).__init__()
        # For the first layer, in_channels is 3 (raw coords as features)
        self.kpconv1 = KPConvEfficient(
            in_channels=3, out_channels=16, num_kernel_points=15, sigma=0.5, k=k
        )
        self.kpconv2 = KPConvEfficient(
            in_channels=16, out_channels=16, num_kernel_points=15, sigma=0.5, k=k
        )
        self.kpconv3 = KPConvEfficient(
            in_channels=16, out_channels=32, num_kernel_points=15, sigma=0.5, k=k
        )
        self.kpconv4 = KPConvEfficient(
            in_channels=32, out_channels=64, num_kernel_points=15, sigma=0.5, k=k
        )
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
        )

    def forward(self, coords):
        """
        Process each point cloud sample individually using only valid (unpadded) points.
        Args:
            coords (Tensor): (B, 3, N) raw point coordinates.
        Returns:
            Tensor: (B, feature_dim) feature vector per sample.
        """
        outputs = []
        B = coords.size(0)
        for i in range(B):
            # Create a mask where valid points have non-zero sum
            valid_mask = (coords[i].abs().sum(dim=0) != 0)
            # Select valid points; shape becomes (1, 3, valid_N)
            valid_coords = coords[i][:, valid_mask].unsqueeze(0)
            # Normalize the valid point cloud
            valid_coords = normalize_point_clouds(valid_coords)

            # Process with KPConv layers using valid_coords
            x = self.kpconv1(valid_coords, valid_coords)  # (1, 16, valid_N)
            x1 = torch.relu(x)
            x = self.kpconv2(valid_coords, x1)  # (1, 16, valid_N)
            x2 = torch.relu(x)
            x = self.kpconv3(valid_coords, x2)  # (1, 32, valid_N)
            x3 = torch.relu(x)
            x = self.kpconv4(valid_coords, x3)  # (1, 64, valid_N)
            x4 = torch.relu(x)

            # Concatenate features and perform global pooling over valid points
            x_cat = torch.cat([x1, x2, x3, x4], dim=1)  # (1, 16+16+32+64, valid_N)
            x_max = torch.max(x_cat, dim=2)[0]
            x_mean = torch.mean(x_cat, dim=2)
            x_pool = torch.cat([x_max, x_mean], dim=1)
            out = self.fc(x_pool)
            outputs.append(out)
        return torch.cat(outputs, dim=0)  # (B, feature_dim)

