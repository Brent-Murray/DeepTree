import torch
import torch.nn as nn
import torch.nn.functional as F


def add_tensors(tensor_list):
    result = torch.zeros_like(tensor_list[0])

    for tensor in tensor_list:
        result += tensor
    return result


def mul_tensors(tensor_list):
    result = torch.ones_like(tensor_list[0])

    for tensor in tensor_list:
        result *= tensor
    return result


def average_tensors(tensor_list):
    result = torch.zeros_like(tensor_list[0])

    for tensor in tensor_list:
        result += tensor

    average_result = result / len(tensor_list)
    return average_result

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
        
        # Weighted output
        self.weighted=weighted
        if weighted:
            # Learnable weights 
            self.weights = nn.Parameter(torch.ones(num_models))
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(num_channels * num_models, num_channels, kernel_size=1)

    def forward(self, model_outputs):
        # model_outputs: list of tensors of shape [B, C, H, W]
        _, _, H, W = model_outputs[0].size()

        if not self.weighted:
            # Apply CSAM to each model output
            csam_outputs = [
                csam(output) for csam, output in zip(self.csam_modules, model_outputs)
            ]

            # Max pooling
            pooled_outputs = [self.max_pool(output) for output in csam_outputs]
            
            concatenated_output = torch.cat(pooled_outputs, dim=1)
        
        
        # Apply CSAM and weight each output
        
        else:
            weighted_outputs = []
            for weight, csam, output in zip(self.weights, self.csam_modules, model_outputs):
                weighted_output = weight * csam(output)
                pooled_output = self.max_pool(weighted_output)
                weighted_outputs.append(pooled_output)

            # Concatenate the outputs along the channel dimension
            concatenated_output = torch.cat(weighted_outputs, dim=1)

        # 1x1 Convolution to mix features and match channel dimensions
        combined_output = self.conv(concatenated_output)

        # Upsampling to original size
        combined_output = F.interpolate(
            combined_output, size=(H, W), mode="bilinear", align_corners=False
        )

        return combined_output
        
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Classify(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classify, self).__init__()
        self.classify = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.classify(x)
    

class JoinConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(JoinConv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = torch.cat(x, dim=1)

        return self.conv(x)
    
class JoinConvBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(JoinConvBN, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels * 0.25), kernel_size=1),
            nn.BatchNorm2d(int(out_channels * 0.25)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                int(out_channels * 0.25), int(out_channels * 0.25), kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(int(out_channels * 0.25)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channels * 0.25), out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return self.conv_bn(x)
    
    
class FusionUNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_models=2, join="multiply"):
        super(FusionUNet, self).__init__()
        # Join Tensor Method
        self.join = join

        # Down Convolutions
        self.double_conv9 = DoubleConv(in_channels, 64)
        self.double_conv1 = DoubleConv(1, 64)
        down_conv1 = DownConv(64, 128)
        down_conv2 = DownConv(128, 256)
        down_conv3 = DownConv(256, 512)
        self.down_convs = nn.ModuleList(
            [down_conv1, down_conv2, down_conv3]
        )
        
        if join == "attention":
            self.attn_1 = AttentionFusionLayer(num_models, 64, weighted=True)
            attn_2 = AttentionFusionLayer(num_models, 128, weighted=True)
            attn_3 = AttentionFusionLayer(num_models, 256, weighted=True)
            attn_4 = AttentionFusionLayer(num_models, 512, weighted=True)
            self.attns = nn.ModuleList([attn_2, attn_3, attn_4])

        # Transpose Convolutions
        self.tran_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.tran_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.tran_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Up Convolutions
        self.up_conv1 = DoubleConv(256 + 256, 256)
        self.up_conv2 = DoubleConv(128 + 128, 128)
        self.up_conv3 = DoubleConv(64 + 64, 64)

        # Classify
        self.classify = Classify(64, num_classes)

    def forward(self, data):
        x = list(data.values())
        feats = []

        # Double Conv
        x = [
            self.double_conv9(xi) if xi.size(1) == 9 else self.double_conv1(xi)
            for xi in x
        ]
        feats.append(self.attn_1(x))
        # Down convolutions
        for i, down_conv in enumerate(self.down_convs):
            x = [down_conv(xi) for xi in x]
            if self.join == "multiply":
                feats.append(mul_tensors(x))
            elif self.join == "add":
                feats.append(add_tensors(x))
            elif self.join == "average":
                feats.append(average_tensors(x))
            elif self.join == "conv":
                join_conv = JoinConv(int(x[0].size(1) * len(x)), int(x[0].size(1))).to(
                    x[0].device
                )
                feats.append(join_conv(x))
            elif self.join == "conv_bn":
                join_conv = JoinConvBN(int(x[0].size(1) * len(x)), int(x[0].size(1))).to(
                    x[0].device
                )
                feats.append(join_conv(x))
            elif self.join == "attention":
                feats.append(self.attns[i](x))
            else:
                raise Exception(f"Join method {self.join} not implemented")

        # Up Convolutions with Transpose Convolutions
        x = self.tran_conv1(feats[3])
        x = torch.cat([x, feats[2]], dim=1)
        x = self.up_conv1(x)

        x = self.tran_conv2(x)
        x = torch.cat([x, feats[1]], dim=1)
        x = self.up_conv2(x)

        x = self.tran_conv3(x)
        x = torch.cat([x, feats[0]], dim=1)
        x = self.up_conv3(x)

        x = self.classify(x)

        return x