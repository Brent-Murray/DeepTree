import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ Helper Functions ------------------
def knn(x, k):
    # x: [B, C, N]
    B, C, N = x.size()
    x_perm = x.permute(0, 2, 1).contiguous()
    inner = -2 * torch.bmm(x_perm, x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1].clone().contiguous()
    return idx


def get_graph_feature(x, k=20, idx=None):
    # x: [B, C, N]
    B, C, N = x.size()
    x = x.contiguous().view(B, -1, N)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    x = x.transpose(2, 1).contiguous()  # [B, N, C]
    feature = x.reshape(B * N, -1)[idx, :].contiguous()
    feature = feature.view(B, N, k, C).contiguous()
    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)
    combined = torch.cat((feature - x, x), dim=3).contiguous()  # [B, N, k, 2C]
    return combined.permute(0, 3, 1, 2).contiguous()  # [B, 2C, N, k]


def normalize_point_clouds(xyz):
    # xyz: [B, 3, N]
    B, C, N = xyz.shape
    centroids = xyz.mean(dim=2, keepdim=True)
    xyz_centered = xyz - centroids
    max_distances = (
        (xyz_centered**2).sum(dim=1).sqrt().max(dim=1, keepdim=True)[0]
    ).unsqueeze(-1)
    xyz_norm = xyz_centered / (max_distances + 1e-8)
    return xyz_norm


def index_points(points, idx):
    # points: [B, N, C], idx: [B, N, k]
    B, N, C = points.shape
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand_as(idx)
    new_points = points[batch_indices, idx, :]
    return new_points


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

    # Fallback: use original interpolation logic.
    if num_layers == 4:
        mid = first_dim + (last_dim - first_dim) // 2
        dims = [first_dim, first_dim, mid, last_dim]
    elif num_layers < 4:
        mid = first_dim + (last_dim - first_dim) // 2
        base = [first_dim, first_dim, mid, last_dim]
        indices = np.linspace(0, len(base) - 1, num_layers)
        dims = [base[int(round(i))] for i in indices]
    else:
        dims = np.linspace(first_dim, last_dim, num_layers, dtype=int).tolist()

    dims = [round_to_multiple(d) for d in dims]
    return dims


def generate_kernel_points(num_points, radius):
    """Generate uniformly distributed kernel points on a sphere using a Fibonacci lattice."""
    points = []
    offset = 2.0 / num_points
    increment = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(num_points):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(max(0, 1 - y * y))
        phi = i * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        points.append([x, y, z])
    points = torch.tensor(points, dtype=torch.float) * radius
    return points


def knn_query(coords, k, exclude_self=True):
    """
    Computes k‑nearest neighbors using torch.cdist.

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


# ------------------ KPConv Layer ------------------
class KPConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_kernel_points=15,
        radius=1.0,
        sigma=0.1,
        k=16,
        bias=True,
    ):
        """
        KPConv layer closer to the original design.

        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
            num_kernel_points (int): Number of kernel points.
            radius (float): Radius used to scale kernel points.
            sigma (float): Gaussian sigma for influence weights.
            k (int): Number of neighbors to query.
            bias (bool): Whether to include a bias term.
        """
        super(KPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernel_points = num_kernel_points
        self.radius = radius
        self.sigma = sigma
        self.k = k

        # Use a deterministic, uniformly distributed set of kernel points.
        kernel_points = generate_kernel_points(num_kernel_points, radius)
        self.register_buffer(
            "kernel_points", kernel_points
        )  # non-learnable fixed kernel points

        # Learnable kernel weights: (num_kernel_points, in_channels, out_channels)
        self.kernel_weights = nn.Parameter(
            torch.randn(num_kernel_points, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, coords, features):
        """
        Args:
            coords (Tensor): (B, 3, N) raw point coordinates.
            features (Tensor): (B, in_channels, N) point features.
        Returns:
            Tensor: (B, out_channels, N) output features.
        """
        B, _, N = coords.shape
        # Ensure using raw coordinates (B, N, 3)
        X = coords.transpose(1, 2).contiguous()
        feat = features.transpose(1, 2).contiguous()

        # k‑NN query based on raw coordinates
        nbr_idx = knn_query(X, self.k, exclude_self=True)  # (B, N, k)
        batch_indices = (
            torch.arange(B, device=X.device).view(B, 1, 1).expand(B, N, self.k)
        )
        neighbor_coords = X[batch_indices, nbr_idx, :]  # (B, N, k, 3)
        neighbor_features = feat[batch_indices, nbr_idx, :]  # (B, N, k, in_channels)

        query_coords = X.unsqueeze(2)  # (B, N, 1, 3)
        diff = neighbor_coords - query_coords  # (B, N, k, 3)

        diff_exp = diff.unsqueeze(3)  # (B, N, k, 1, 3)
        kp = self.kernel_points.view(
            1, 1, 1, self.num_kernel_points, 3
        )  # (1, 1, 1, K, 3)
        diff_rel = diff_exp - kp  # (B, N, k, K, 3)
        sq_dist = (diff_rel**2).sum(dim=-1)  # (B, N, k, K)
        influence = torch.exp(-sq_dist / (self.sigma**2))  # (B, N, k, K)

        aggregated = torch.einsum(
            "bnki,bnkc->bnic", influence, neighbor_features
        )  # (B, N, K, in_channels)
        out = torch.einsum(
            "bnki,kio->bno", aggregated, self.kernel_weights
        )  # (B, N, out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out.transpose(1, 2)  # (B, out_channels, N)


# ------------------ EdgeConv Layer ------------------
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


class EdgeConv(nn.Module):
    def __init__(self, input_dim, output_dim, k=20):
        super(EdgeConv, self).__init__()
        self.conv = Conv2dLayerNormReLU(input_dim, output_dim)
        self.k = k

    def forward(self, x):
        edge = get_graph_feature(x, self.k)
        x = self.conv(edge).max(dim=-1)[0]

        return x


# ------------------ Point Transformer Layer ------------------
class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=16, num_heads=4):
        super(PointTransformerLayer, self).__init__()
        assert (
            out_channels % num_heads == 0
        ), "out_channels must be divisible by num_heads"
        self.k = k
        self.num_heads = num_heads
        self.dim_per_head = out_channels // num_heads

        # Linear projections for multi-head attention
        self.linear_q = nn.Linear(in_channels, out_channels)
        self.linear_k = nn.Linear(in_channels, out_channels)
        self.linear_v = nn.Linear(in_channels, out_channels)

        # Positional encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)
        )

        self.fc = nn.Linear(out_channels, out_channels)
        self.softmax = nn.Softmax(dim=-1)

        # Shortcut projection if needed
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def forward(self, x, pos):
        B, N, _ = x.shape
        idx = knn(pos.transpose(1, 2).contiguous(), k=self.k)
        x_neighbors = index_points(x, idx)
        pos_neighbors = index_points(pos, idx)
        pos_diff = pos.unsqueeze(2) - pos_neighbors
        pos_enc = self.pos_mlp(pos_diff)

        q = self.linear_q(x)
        k_proj = self.linear_k(x_neighbors)
        v = self.linear_v(x_neighbors)

        q = q.view(B, N, self.num_heads, self.dim_per_head)
        k_proj = k_proj.view(B, N, self.k, self.num_heads, self.dim_per_head).permute(
            0, 1, 3, 2, 4
        )
        v = v.view(B, N, self.k, self.num_heads, self.dim_per_head).permute(
            0, 1, 3, 2, 4
        )
        q = q.unsqueeze(3)
        pos_enc = pos_enc.view(B, N, self.k, self.num_heads, self.dim_per_head).permute(
            0, 1, 3, 2, 4
        )

        attn = q - k_proj + pos_enc
        attn = attn.mean(dim=-1)
        attn = self.softmax(attn)

        agg = torch.sum(attn.unsqueeze(-1) * (v + pos_enc), dim=3)
        agg = agg.view(B, N, -1)
        out = self.fc(agg)

        shortcut = self.shortcut(x) if self.shortcut is not None else x
        return shortcut + out


# ------------------ Hybrid Layer ------------------
class HybridLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        k_edge=20,
        k_pt=16,
        k_kp=16,
        num_heads=4,
        num_kernel_points=15,
        radius=1.0,
        sigma=0.1,
    ):
        super(HybridLayer, self).__init__()
        # For EdgeConv, note that the input dimension is expected to be doubled.
        self.edgeconv = EdgeConv(in_channels * 2, out_channels, k=k_edge)
        # self.kpconv = KPConv(
        #     in_channels, out_channels, num_kernel_points, radius, sigma, k=k_kp
        # )
        self.point_transformer = PointTransformerLayer(
            in_channels, out_channels, k=k_pt, num_heads=num_heads
        )
        # Combine the three branch outputs (each of dimension out_channels)
        # self.proj = nn.Linear(3 * out_channels, out_channels)
        self.proj = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x, pos):
        # x: (B, N, in_channels), pos: (B, N, 3)
        # PointTransformer branch: (B, N, out_channels)
        pt_out = self.point_transformer(x, pos)
        # KPConv branch: expect coords (B, 3, N) and features (B, in_channels, N)
        # kp_out = self.kpconv(
        #     pos.transpose(1, 2).contiguous(), x.transpose(1, 2).contiguous()
        # )
        # kp_out = kp_out.transpose(1, 2).contiguous()
        # EdgeConv branch: expects x: (B, in_channels, N)
        ec_out = self.edgeconv(x.transpose(1, 2).contiguous())
        ec_out = ec_out.transpose(1, 2).contiguous()

        # Concatenate outputs from all three branches and project.
        # out = torch.cat([pt_out, kp_out, ec_out], dim=-1)
        out = torch.cat([pt_out, ec_out], dim=-1)
        out = self.proj(out)
        return out


# ------------------ Point Feature Extractor ------------------
class PointExtractor(nn.Module):
    def __init__(
        self, first_dim=16, last_dim=64, layers=4, extractor="pointtransformer"
    ):
        super(PointExtractor, self).__init__()
        self.extractor = extractor
        dims = get_layer_dims(first_dim, last_dim, layers)
        layers = []
        in_dim = 3
        if extractor in ["pointtransformer", "hybrid"]:
            self.input_linear = nn.Linear(3, first_dim)
            in_dim = first_dim
        for out_dim in dims:
            if extractor == "pointtransformer":
                layers.append(PointTransformerLayer(in_dim, out_dim, k=16))
            elif extractor == "edgeconv":
                layers.append(EdgeConv(in_dim * 2, out_dim, k=20))
            elif extractor == "kpconv":
                layers.append(KPConv(in_dim, out_dim, k=16))
            elif extractor == "hybrid":
                layers.append(HybridLayer(in_dim, out_dim))
            else:
                raise ValueError(
                    f"Unsupported extractor: {extractor}. Please use 'pointtransformer', 'edgeconv', 'kpconv' or 'hybrid'."
                )
            in_dim = out_dim
        self.extractor_layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = []
        B = x.size(0)
        for i in range(B):
            valid_mask = x[i].abs().sum(dim=0) != 0
            valid_x = x[i][:, valid_mask].unsqueeze(0)
            valid_x = normalize_point_clouds(valid_x)

            # Run Point Extractor
            x_out = []
            if self.extractor in ["pointtransformer", "hybrid"]:
                # For these extractors, use an input linear transformation.
                pos = valid_x.transpose(1, 2).contiguous()
                valid_x = F.relu(self.input_linear(pos))
            if self.extractor == "kpconv":
                pos = valid_x

            for layer in self.extractor_layers:
                if self.extractor in ["pointtransformer", "hybrid"]:
                    valid_x = layer(valid_x, pos)
                elif self.extractor == "edgeconv":
                    valid_x = layer(valid_x)
                elif self.extractor == "kpconv":
                    valid_x = layer(pos, valid_x)

                x_out.append(valid_x)

            if self.extractor in ["pointtransformer", "hybrid"]:
                x_cat = torch.cat(x_out, dim=2)
                max_pool = torch.max(x_cat, dim=1)[0]
            elif self.extractor == "edgeconv":
                x_cat = torch.cat(x_out, dim=1)
                max_pool = torch.max(x_cat, dim=-1)[0]
            elif self.extractor == "kpconv":
                x_cat = torch.cat(x_out, dim=1)
                max_pool = torch.max(x_cat, dim=2)[0]

            outputs.append(max_pool)

        return torch.cat(outputs, dim=0)
