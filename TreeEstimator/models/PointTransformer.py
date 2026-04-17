import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ------------------ K-Nearest Neighbours ------------------
def knn(x, k):
    # x: [B, C, N]
    B, C, N = x.size()
    x_perm = x.permute(0, 2, 1).contiguous()
    inner = -2 * torch.bmm(x_perm, x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1].clone().contiguous()
    return idx


# ------------------ Gather Neighbor Points ------------------
def index_points(points, idx):
    # points: [B, N, C], idx: [B, N, k]
    B, N, C = points.shape
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand_as(idx)
    new_points = points[batch_indices, idx, :]
    return new_points


# ------------------ Revised Point Transformer Layer ------------------
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


# ------------------ Point Transformer Feature Extractor ------------------
class PointTransformerFeatureExtractor(nn.Module):
    def __init__(self, k=16, feature_dim=256, num_heads=4):
        super(PointTransformerFeatureExtractor, self).__init__()
        # Initial Embedding
        self.input_linear = nn.Linear(3, 16)

        # Four transformer layers to mimic DGCNN
        self.pt_layer1 = PointTransformerLayer(16, 16, k=k, num_heads=num_heads)
        self.pt_layer2 = PointTransformerLayer(16, 16, k=k, num_heads=num_heads)
        self.pt_layer3 = PointTransformerLayer(16, 32, k=k, num_heads=num_heads)
        self.pt_layer4 = PointTransformerLayer(32, 64, k=k, num_heads=num_heads)

    def forward(self, x):
        # x: [B, 3, N]
        outputs = []
        B = x.size(0)
        for i in range(B):
            valid_mask = x[i].abs().sum(dim=0) != 0
            valid_x = x[i][:, valid_mask].unsqueeze(0)
            valid_x = normalize_point_clouds(valid_x)
            pos = valid_x.transpose(1, 2).contiguous()  # [B, N, 3]

            x_feat = F.relu(self.input_linear(pos))  # [B, N, 16]

            x1 = F.relu(self.pt_layer1(x_feat, pos))  # [B, N, 16]
            x2 = F.relu(self.pt_layer2(x1, pos))  # [B, N, 16]
            x3 = F.relu(self.pt_layer3(x2, pos))  # [B, N, 32]
            x4 = F.relu(self.pt_layer4(x3, pos))  # [B, N, 64]

            x_cat = torch.cat([x1, x2, x3, x4], dim=2)

            max_pool = torch.max(x_cat, dim=1)[0]  # [B, 128]
            outputs.append(max_pool)

        return torch.cat(outputs, dim=0)
