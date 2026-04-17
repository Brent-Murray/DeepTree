import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ Helper Functions ------------------
def knn(x, k):
    # x: [B, C, N]
    B, C, N = x.size()
    x_perm = x.permute(0, 2, 1).contiguous()  # [B, N, C]
    inner = -2 * torch.bmm(x_perm, x)  # [B, N, N]
    xx = torch.sum(x**2, dim=1, keepdim=True)  # [B, 1, N]
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
    batch_indices = (
        torch.arange(B, device=points.device)
        .view(B, 1, 1)
        .expand(B, idx.shape[1], idx.shape[2])
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def get_pt_dims(num_layers):
    # If exactly 4 layers, use the desired pattern.
    if num_layers == 4:
        return [16, 16, 32, 64]
    # If fewer than 4, sample from the base pattern.
    elif num_layers < 4:
        base = [16, 16, 32, 64]
        indices = np.linspace(0, len(base) - 1, num_layers)
        return [base[int(round(i))] for i in indices]
    # For more than 4 layers, linearly interpolate between 8 and 64.
    else:
        return np.linspace(8, 64, num_layers, dtype=int).tolist()


# ------------------ Modified DGCNN Extractor ------------------
class ModifiedDGCNNExtractor(nn.Module):
    def __init__(self, k=20):
        super(ModifiedDGCNNExtractor, self).__init__()
        self.k = k
        self.conv1 = Conv2dLayerNormReLU(6, 16)
        self.conv2 = Conv2dLayerNormReLU(16 * 2, 16)
        self.conv3 = Conv2dLayerNormReLU(16 * 2, 32)
        self.conv4 = Conv2dLayerNormReLU(32 * 2, 64)
        self.align1 = nn.Conv1d(16, 32, 1, bias=False)
        self.align2 = nn.Conv1d(32, 64, 1, bias=False)

    def forward(self, x):
        # x: [B, 3, N]
        # x = normalize_point_clouds(x)
        # First edge: get graph feature then conv1
        x_feature = get_graph_feature(x, k=self.k)  # [B, 6, N, k]
        x = self.conv1(x_feature)  # -> [B, 8, N, k]
        x1 = x.max(dim=-1)[0]  # [B, 8, N]

        # Second edge:
        x_feature = get_graph_feature(x1, k=self.k)
        x = self.conv2(x_feature)  # [B, 8, N, k]
        x2 = x.max(dim=-1)[0]  # [B, 8, N]
        x2 = x2 + x1

        # Third edge:
        x_feature = get_graph_feature(x2, k=self.k)
        x = self.conv3(x_feature)  # [B, 16, N, k]
        x3 = x.max(dim=-1)[0]  # [B, 16, N]
        x3 = x3 + self.align1(x2)

        # Fourth edge:
        x_feature = get_graph_feature(x3, k=self.k)
        x = self.conv4(x_feature)  # [B, 32, N, k]
        x4 = x.max(dim=-1)[0]  # [B, 32, N]
        x4 = x4 + self.align2(x3)

        # Concatenate per-point features (channels: 64)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)  # [B, 64, N]
        return x_cat


# ------------------ Conv2d + LayerNorm + LeakyReLU Block ------------------
class Conv2dLayerNormReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, bias=False, negative_slope=0.2
    ):
        super(Conv2dLayerNormReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        self.ln = nn.LayerNorm(out_channels)  # Layer norm over channel dim
        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, out_channels)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # (B, out_channels, H, W)
        return self.act(x)


# ------------------ Modified KPConv Extractor ------------------
class ModifiedKPConvExtractor(nn.Module):
    def __init__(self, k=16):
        super(ModifiedKPConvExtractor, self).__init__()
        self.k = k
        self.kpconv1 = KPConvEfficient(
            in_channels=128, out_channels=16, num_kernel_points=15, sigma=0.5, k=k
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

    def forward(self, coords, features):
        # coords: [B, 3, N]
        # coords = normalize_point_clouds(coords)
        x = self.kpconv1(coords, features)  # (B, 8, N)
        x1 = F.relu(x)
        x = self.kpconv2(coords, x1)  # (B, 8, N)
        x2 = F.relu(x)
        x = self.kpconv3(coords, x2)  # (B, 16, N)
        x3 = F.relu(x)
        x = self.kpconv4(coords, x3)  # (B, 32, N)
        x4 = F.relu(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 64, N) # 128
        return x_cat


# ------------------ Provided KPConvEfficient ------------------
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
        super(KPConvEfficient, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernel_points = num_kernel_points
        self.sigma = sigma
        self.k = k
        kernel_points = torch.randn(num_kernel_points, 3)
        kernel_points = kernel_points / kernel_points.norm(dim=1, keepdim=True)
        self.register_parameter(
            "kernel_points", nn.Parameter(kernel_points, requires_grad=False)
        )
        self.kernel_weights = nn.Parameter(
            torch.randn(num_kernel_points, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, coords, features):
        # coords: [B, 3, N], features: [B, in_channels, N]
        B, _, N = coords.shape
        X = coords.transpose(1, 2).contiguous()  # (B, N, 3)
        feat = features.transpose(1, 2).contiguous()  # (B, N, in_channels)
        nbr_idx = knn(X.transpose(1, 2).contiguous(), self.k)  # (B, N, k)
        batch_indices = (
            torch.arange(B, device=X.device).view(B, 1, 1).expand(B, N, self.k)
        )
        neighbor_coords = X[batch_indices, nbr_idx, :]  # (B, N, k, 3)
        query = X.unsqueeze(2)  # (B, N, 1, 3)
        diff = neighbor_coords - query  # (B, N, k, 3)
        diff_exp = diff.unsqueeze(3)  # (B, N, k, 1, 3)
        kp = self.kernel_points.view(1, 1, 1, self.num_kernel_points, 3)
        diff_rel = diff_exp - kp  # (B, N, k, K, 3)
        sq_dist = (diff_rel**2).sum(dim=-1)  # (B, N, k, K)
        influence = torch.exp(-sq_dist / (self.sigma**2))  # (B, N, k, K)
        neighbor_features = feat[batch_indices, nbr_idx, :]  # (B, N, k, in_channels)
        aggregated = torch.einsum(
            "bnki,bnkc->bnic", influence, neighbor_features
        )  # (B, N, K, in_channels)
        out = torch.einsum(
            "bnki,kio->bno", aggregated, self.kernel_weights
        )  # (B, N, out_channels)
        if self.bias is not None:
            out = out + self.bias
        return out.transpose(1, 2)  # (B, out_channels, N)


# ------------------ Point Transformer Layer ------------------
class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super(PointTransformerLayer, self).__init__()
        self.k = k
        self.linear_center = nn.Linear(in_channels, out_channels)
        self.linear_neighbor = nn.Linear(in_channels, out_channels)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.attn_score = nn.Linear(out_channels, 1)

    def forward(self, x, pos):
        # x: [B, N, in_channels], pos: [B, N, 3]
        B, N, _ = x.shape
        idx = knn(pos.transpose(1, 2).contiguous(), self.k)  # (B, N, k)
        x_neighbors = index_points(x, idx)  # (B, N, k, in_channels)
        pos_neighbors = index_points(pos, idx)  # (B, N, k, 3)
        pos_diff = pos.unsqueeze(2) - pos_neighbors  # (B, N, k, 3)
        pos_enc = self.pos_mlp(pos_diff)  # (B, N, k, out_channels)
        feat_center = self.linear_center(x)  # (B, N, out_channels)
        feat_neighbors = self.linear_neighbor(x_neighbors)  # (B, N, k, out_channels)
        attn = self.attn_mlp(
            feat_center.unsqueeze(2) - feat_neighbors + pos_enc
        )  # (B, N, k, out_channels)
        attn = self.attn_score(attn).squeeze(-1)  # (B, N, k)
        attn = F.softmax(attn, dim=-1)
        aggregated = torch.sum(
            attn.unsqueeze(-1) * (feat_neighbors + pos_enc), dim=2
        )  # (B, N, out_channels)
        out = feat_center + aggregated  # (B, N, out_channels)
        return out


# ------------------ Hybrid Parallel-Sequential Extractor ------------------
class HybridParallelSequentialExtractor(nn.Module):
    def __init__(self, fusion_dim=256, pt_layers=4, pt_k=16):
        super(HybridParallelSequentialExtractor, self).__init__()
        # Parallel branches (each returns per-point features with 128 channels)
        self.dgcnn_branch = ModifiedDGCNNExtractor(k=20)
        self.kpconv_branch = ModifiedKPConvExtractor(k=16)
        # Concatenated feature: 64 + 64 = 128 channels.
        self.fusion_conv = nn.Conv1d(256, 256, kernel_size=1)
        # Sequential processing: stack of PointTransformerLayer.
        pt_dims = get_pt_dims(pt_layers)
        layers = []
        in_dim = 256
        for out_dim in pt_dims:
            layers.append(PointTransformerLayer(in_dim, out_dim, k=pt_k))
            in_dim = out_dim
        self.pt_layers = nn.ModuleList(layers)

        # Global pooling then final fully-connected layer.
        self.fc = nn.Linear(640, fusion_dim)

    def forward(self, x):
        # x: [B, 3, N]
        # Parallel branches
        outputs = []
        B = x.size(0)
        for i in range(B):
            valid_mask = x[i].abs().sum(dim=0) != 0
            valid_x = x[i][:, valid_mask].unsqueeze(0)
            valid_x = normalize_point_clouds(valid_x)
            feat_dgcnn = self.dgcnn_branch(valid_x)  # [B, 64, N]
            feat_kpconv = self.kpconv_branch(valid_x, feat_dgcnn)  # [B, 64, N]
            # Concatenate along channel dimension: [B, 128, N]
            fused = torch.cat([feat_dgcnn, feat_kpconv], dim=1)
            # Fuse via a 1x1 convolution: [B, 64, N]
            fused = self.fusion_conv(fused)
            # Prepare for sequential processing: transpose to [B, N, 64]
            fused = fused.transpose(1, 2).contiguous()
            # Use the normalized coordinates as positions.
            pos = valid_x.transpose(1, 2).contiguous()  # [B, N, 3]
            # Sequential processing through PointTransformer layers.
            for layer in self.pt_layers:
                fused = layer(fused, pos)
            # Global pooling over N:
            max_pool = torch.max(fused, dim=1)[0]  # [B, 64]
            mean_pool = torch.mean(fused, dim=1)  # [B, 64]
            global_feat_pt = torch.cat([max_pool, mean_pool], dim=1)  # [B, 64*2]

            max_pool = torch.max(feat_dgcnn, dim=2)[0]
            mean_pool = torch.mean(feat_dgcnn, dim=2)
            global_feat_dgcnn = torch.cat([max_pool, mean_pool], dim=1)  # [B, 64*2]

            max_pool = torch.max(feat_kpconv, dim=2)[0]
            mean_pool = torch.mean(feat_kpconv, dim=2)
            global_feat_kpconv = torch.cat([max_pool, mean_pool], dim=1)  # [B, 64*2]

            feats = torch.cat(
                [global_feat_pt, global_feat_dgcnn, global_feat_kpconv], dim=1
            )  # [B, 256*3]

            out = self.fc(feats)  # [B, fusion_dim]
            outputs.append(out)

        return torch.cat(outputs, dim=0)
