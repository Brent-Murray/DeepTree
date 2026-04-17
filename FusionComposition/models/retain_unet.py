# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def add_tensors(tensor_list):
#     result = torch.zeros_like(tensor_list[0])

#     for tensor in tensor_list:
#         result += tensor
#     return result

# class CSAM(nn.Module):
#     def __init__(self, in_channels):
#         super(CSAM, self).__init__()
#         # Channel attention
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
#         )
#         self.sigmoid_channel = nn.Sigmoid()

#         # Spatial attention
#         self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.sigmoid_spatial = nn.Sigmoid()

#     def forward(self, x):
#         # Channel attention
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         channel_att = self.sigmoid_channel(avg_out + max_out)

#         # Apply channel attention
#         x = channel_att * x

#         # Spatial attention
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         spatial_att = torch.cat([avg_out, max_out], dim=1)
#         spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_att))

#         # Apply spatial attention
#         x = spatial_att * x

#         return x

# class AttentionFusionLayer(nn.Module):
#     def __init__(self, num_models, num_channels, weighted=False):
#         super(AttentionFusionLayer, self).__init__()
#         self.csam_modules = nn.ModuleList(
#             [CSAM(num_channels) for _ in range(num_models)]
#         )

#         self.weighted = weighted
#         if weighted:
#             self.weights = nn.Parameter(
#                 torch.ones(num_models, 1, 1, 1)
#             )  # Ensuring broadcasting works properly
#         self.conv = nn.Conv2d(num_channels * num_models, num_channels, kernel_size=1)

#     def forward(self, model_outputs):
#         # Apply CSAM to each model output and optionally apply weights
#         if not self.weighted:
#             combined_output = torch.cat(
#                 [
#                     csam(output)
#                     for csam, output in zip(self.csam_modules, model_outputs)
#                 ],
#                 dim=1,
#             )
#         else:
#             combined_output = torch.cat(
#                 [
#                     weight * csam(output)
#                     for weight, csam, output in zip(
#                         self.weights, self.csam_modules, model_outputs
#                     )
#                 ],
#                 dim=1,
#             )

#         # 1x1 Convolution to mix features and match channel dimensions
#         combined_output = self.conv(combined_output)
#         return combined_output

# class DualConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DualConv, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x)

#         return add_tensors([x1, x2])

# class ReduceConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ReduceConv, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv1(x)

# class Classify(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(Classify, self).__init__()
#         self.classify = nn.Sequential(
#             nn.Conv2d(in_channels, num_classes, kernel_size=1)
#         )

#     def forward(self, x):
#         return self.classify(x)

# class RetFuseNet(nn.Module):
#     def __init__(self, in_channels, num_classes, num_models=2, join="attention"):
#         super(RetFuseNet, self).__init__()
#         self.join = join
#         self.in_channels = in_channels

#         # Dual Convolutions
#         self.base_conv = nn.Conv2d(1, 9, kernel_size=1)
#         dual_conv1 = DualConv(in_channels, 64)
#         dual_conv2 = DualConv(64, 128)
#         dual_conv3 = DualConv(128, 256)
#         dual_conv4 = DualConv(256, 512)
#         self.dual_convs = nn.ModuleList(
#             [dual_conv1, dual_conv2, dual_conv3, dual_conv4]
#         )

#         # Fusion
#         if join == "attention":
#             attn_1 = AttentionFusionLayer(num_models, 64, weighted=True)
#             attn_2 = AttentionFusionLayer(num_models, 128, weighted=True)
#             attn_3 = AttentionFusionLayer(num_models, 256, weighted=True)
#             attn_4 = AttentionFusionLayer(num_models, 512, weighted=True)
#             self.attns = nn.ModuleList([attn_1, attn_2, attn_3, attn_4])

#         # Reduce Convolutions
#         self.red_conv1 = ReduceConv(512, 256)
#         self.red_conv2 = ReduceConv(512, 128)
#         self.red_conv3 = ReduceConv(256, 64)
#         self.red_conv4 = ReduceConv(128, 32)
#         self.classify = Classify(32, num_classes)

#     def forward(self, data):
#         x = list(data.values())
#         feats = []

#         x = [self.base_conv(xi) if xi.size(1) != 9 else xi for xi in x]

#         for i, dual_conv in enumerate(self.dual_convs):
#             x = [dual_conv(xi) for xi in x]
#             if self.join == "attention":
#                 feats.append(self.attns[i](x))
#             else:
#                 raise Exception(f"Join method {self.join} not implemented")

#         x = self.red_conv1(feats[3])  # 512 -> 256
#         x = torch.cat([x, feats[2]], dim=1)  # 256 + 256
#         x = self.red_conv2(x)  # 256 + 256 -> 128
#         x = torch.cat([x, feats[1]], dim=1)  # 128 + 128
#         x = self.red_conv3(x)  # 128 + 128 -> 64
#         x = torch.cat([x, feats[0]], dim=1)  # 64 + 64
#         x = self.red_conv4(x)  # 64 + 64 -> 32
#         x = self.classify(x)  # 32 -> num_classes

#         return x



import torch
import torch.nn as nn
import torch.nn.functional as F

def add_tensors(tensor_list):
    result = torch.zeros_like(tensor_list[0])

    for tensor in tensor_list:
        result += tensor
    return result

class CSAM(nn.Module):
    def __init__(self, in_channels):
        super(CSAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)

        # Apply channel attention
        x = channel_att * x

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_att))

        # Apply spatial attention
        x = spatial_att * x

        return x

class AttentionFusionLayer(nn.Module):
    def __init__(self, num_models, num_channels, weighted=False):
        super(AttentionFusionLayer, self).__init__()
        self.csam_modules = nn.ModuleList(
            [CSAM(num_channels) for _ in range(num_models)]
        )

        self.weighted = weighted
        if weighted:
            self.weights = nn.Parameter(
                torch.ones(num_models, 1, 1, 1)
            )  # Ensuring broadcasting works properly
        self.conv = nn.Conv2d(num_channels * num_models, num_channels, kernel_size=1)

    def forward(self, model_outputs):
        # Apply CSAM to each model output and optionally apply weights
        if not self.weighted:
            combined_output = torch.cat(
                [
                    csam(output)
                    for csam, output in zip(self.csam_modules, model_outputs)
                ],
                dim=1,
            )
        else:
            combined_output = torch.cat(
                [
                    weight * csam(output)
                    for weight, csam, output in zip(
                        self.weights, self.csam_modules, model_outputs
                    )
                ],
                dim=1,
            )

        # 1x1 Convolution to mix features and match channel dimensions
        combined_output = self.conv(combined_output)
        return combined_output

class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        # x2 = self.conv2(x)
        x2 = self.conv2(x1)

        return add_tensors([x1, x2])

class ReduceConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReduceConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return add_tensors([x1, x2])

class Classify(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classify, self).__init__()
        self.classify = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.classify(x)
    
class WeightedConcat(nn.Module):
    def __init__(self, num_channels):
        super(WeightedConcat, self).__init__()
        # Initialize the weights for each channel
        self.weights = nn.Parameter(torch.rand(num_channels))

    def forward(self, tensor1, tensor2):
        # Ensure the weights are normalized such that weight1 + weight2 = 1 for each channel
        weight1 = self.weights / (self.weights + 1.0)
        weight2 = 1.0 - weight1
        
        # Apply weights to tensors
        weighted_tensor1 = tensor1 * weight1.view(1, -1, 1, 1)  # Reshape weights to match tensor dimensions [B, C, H, W]
        weighted_tensor2 = tensor2 * weight2.view(1, -1, 1, 1)

        # Concatenate along the channel axis
        result = torch.cat([weighted_tensor1, weighted_tensor2], dim=1)
        return result

class RetFuseNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_models=2, join="attention"):
        super(RetFuseNet, self).__init__()
        self.join = join
        self.in_channels = in_channels

        # Dual Convolutions
        self.base_conv = nn.Conv2d(1, 9, kernel_size=1)
        dual_conv1 = DualConv(in_channels, 64)
        dual_conv1a = DualConv(in_channels * 2, 64)
        dual_conv2 = DualConv(64, 128)
        dual_conv3 = DualConv(128, 256)
        dual_conv4 = DualConv(256, 512)
        self.dual_convs1 = nn.ModuleList(
            [dual_conv1, dual_conv2, dual_conv3, dual_conv4]
        )
        self.dual_convs2 = nn.ModuleList(
            [dual_conv1a, dual_conv2, dual_conv3, dual_conv4]
        )

        # Fusion
        if join == "attention":
            attn_1 = AttentionFusionLayer(num_models, 64, weighted=True)
            attn_2 = AttentionFusionLayer(num_models, 128, weighted=True)
            attn_3 = AttentionFusionLayer(num_models, 256, weighted=True)
            attn_4 = AttentionFusionLayer(num_models, 512, weighted=True)
            self.attns = nn.ModuleList([attn_1, attn_2, attn_3, attn_4])

        # Reduce Convolutions
        self.red_conv1 = ReduceConv(512, 256)
        self.red_conv2 = ReduceConv(512, 128)
        self.red_conv3 = ReduceConv(256, 64)
        self.red_conv4 = ReduceConv(128, 32)
        self.classify = Classify(32, num_classes)

        self.w_concat = WeightedConcat(num_classes)

    def forward(self, data):
        x = list(data.values())
        x2 = x[-1]
        x = x[:-1]
        
        feats = []

        x = [self.base_conv(xi) if xi.size(1) != 9 else xi for xi in x]

        for i, dual_conv in enumerate(self.dual_convs1):
            x = [dual_conv(xi) for xi in x]
            if self.join == "attention":
                feats.append(self.attns[i](x))
            else:
                raise Exception(f"Join method {self.join} not implemented")

        x = self.red_conv1(feats[3])  # 512 -> 256
        x = torch.cat([x, feats[2]], dim=1)  # 256 + 256
        x = self.red_conv2(x)  # 256 + 256 -> 128
        x = torch.cat([x, feats[1]], dim=1)  # 128 + 128
        x = self.red_conv3(x)  # 128 + 128 -> 64
        x = torch.cat([x, feats[0]], dim=1)  # 64 + 64
        x = self.red_conv4(x)  # 64 + 64 -> 32
        x = self.classify(x)  # 32 -> num_classes

        x = self.w_concat(x, x2)
        # x = torch.cat([x, x2], dim=1)
        
        feats = []
        for dual_conv in self.dual_convs2:
            x = dual_conv(x)
            feats.append(x)

        x = self.red_conv1(feats[3])  # 512 -> 256
        x = torch.cat([x, feats[2]], dim=1)  # 256 + 256
        x = self.red_conv2(x)  # 256 + 256 -> 128
        x = torch.cat([x, feats[1]], dim=1)  # 128 + 128
        x = self.red_conv3(x)  # 128 + 128 -> 64
        x = torch.cat([x, feats[0]], dim=1)  # 64 + 64
        x = self.red_conv4(x)  # 64 + 64 -> 32
        x = self.classify(x)  # 32 -> num_classes

        return x

