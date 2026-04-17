import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ------------------ Normalize Point Clouds ------------------
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


# ------------------ Gather Neighbor Points ------------------
def index_points(points, idx):
    # points: [B, N, C], idx: [B, N, k]
    B, N, C = points.shape
    # Create a batch index tensor
    batch_indices = (
        torch.arange(B, device=points.device)
        .view(B, 1, 1)
        .expand(B, idx.shape[1], idx.shape[2])
    )
    new_points = points[batch_indices, idx, :]
    return new_points


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
        # x: [B, N, C], pos: [B, N, 3]
        B, N, _ = x.shape
        # Compute kNN based on position
        idx = knn(pos.permute(0, 2, 1).contiguous(), k=self.k)  # [B, N, k]
        x_neighbors = index_points(x, idx)  # [B, N, k, C]
        pos_neighbors = index_points(pos, idx)  # [B, N, k, 3]
        pos_diff = pos.unsqueeze(2) - pos_neighbors  # [B, N, k, 3]
        pos_enc = self.pos_mlp(pos_diff)  # [B, N, k, out_channels]

        feat_center = self.linear_center(x)  # [B, N, out_channels]
        feat_neighbors = self.linear_neighbor(x_neighbors)  # [B, N, k, out_channels]

        # Compute attention scores
        attn = self.attn_mlp(
            feat_center.unsqueeze(2) - feat_neighbors + pos_enc
        )  # [B, N, k, out_channels]
        attn = self.attn_score(attn).squeeze(-1)  # [B, N, k]
        attn = F.softmax(attn, dim=-1)  # normalize over k

        # Aggregate neighbor features
        aggregated = torch.sum(
            attn.unsqueeze(-1) * (feat_neighbors + pos_enc), dim=2
        )  # [B, N, out_channels]
        out = feat_center + aggregated  # residual connection
        return out


# ------------------ Point Transformer Feature Extractor ------------------
class PointTransformerFeatureExtractor(nn.Module):
    def __init__(self, k=16, feature_dim=256):
        super(PointTransformerFeatureExtractor, self).__init__()
        self.k = k
        # Initial embedding: project 3D coords to a low-dim feature space.
        self.input_mlp = nn.Linear(3, 16)
        self.pt_layer1 = PointTransformerLayer(16, 16, k)
        self.pt_layer2 = PointTransformerLayer(16, 32, k)
        self.pt_layer3 = PointTransformerLayer(32, 64, k)
        self.pt_layer4 = PointTransformerLayer(64, 128, k)
        # Fully connected head: maps aggregated features to feature_dim.
        self.fc = nn.Sequential(nn.Linear((16 + 32 + 64 + 128) * 2, feature_dim))

    def forward(self, x):
        # x: [B, 3, N]
        outputs = []
        B = x.size(0)
        for i in range(B):
            valid_mask = x[i].abs().sum(dim=0) != 0
            valid_x = x[i][:, valid_mask].unsqueeze(0)
            valid_x = normalize_point_clouds(valid_x)
            pos = valid_x.transpose(1, 2).contiguous()  # [B, N, 3]
            x_feat = self.input_mlp(pos)  # [B, N, 16]

            out1 = self.pt_layer1(x_feat, pos)  # [B, N, 16]
            out2 = self.pt_layer2(out1, pos)  # [B, N, 32]
            # out2 = out2 + out1  # residual connection
            out3 = self.pt_layer3(out2, pos)  # [B, N, 64]
            # out3 = out3 + out2  # residual connection
            out4 = self.pt_layer4(out3, pos)  # [B, N, 128]
            # out4 = out4 + out3  # residual connection

            # Concatenate features from all layers
            concat_features = torch.cat([out1, out2, out3, out4], dim=-1)  # [B, N, 240]
            # Global pooling: both max and mean
            max_pool = torch.max(concat_features, dim=1)[0]  # [B, 240]
            mean_pool = torch.mean(concat_features, dim=1)  # [B, 240]
            global_feature = torch.cat([max_pool, mean_pool], dim=1)  # [B, 480]

            final_feature = self.fc(global_feature)  # [B, feature_dim]
            outputs.append(final_feature)
        return torch.cat(outputs, dim=0)