import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, num_models, num_channels):
        super(AttentionFusionLayer, self).__init__()
        self.csam_modules = nn.ModuleList(
            [CSAM(num_channels) for _ in range(num_models)]
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(num_channels * num_models, num_channels, kernel_size=1)

    def forward(self, model_outputs):
        # model_outputs: list of tensors of shape [B, C, H, W]

        # Apply CSAM to each model output
        csam_outputs = [
            csam(output) for csam, output in zip(self.csam_modules, model_outputs)
        ]

        # Max pooling
        pooled_outputs = [self.max_pool(output) for output in csam_outputs]

        # Concatenate the outputs along the channel dimension
        concatenated_output = torch.cat(pooled_outputs, dim=1)

        # 1x1 Convolution to mix features and match channel dimensions
        combined_output = self.conv(concatenated_output)

        # Upsampling to original size
        combined_output = F.interpolate(
            combined_output, size=(128, 128), mode="bilinear", align_corners=False
        )

        return combined_output

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), # Changed kernel to 1 / removed padding
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), # Changed kernel to 1  / removed padding
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


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class Classify(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classify, self).__init__()
        self.classify = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.classify(x)
    
class EnsambleUNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_models=4, join="concat"):
        super(EnsambleUNet, self).__init__()

        self.join = join

        # Down Convolutions
        self.down_conv1 = DoubleConv(in_channels, 64)
        self.down_conv2 = DownConv(64, 128)
        self.down_conv3 = DownConv(128, 256)
        self.down_conv4 = DownConv(256, 512)

        # Transpose Convolutions
        self.tran_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.tran_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.tran_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Up Convolutions
        self.up_conv1 = DoubleConv(256 + 256, 256)
        self.up_conv2 = DoubleConv(128 + 128, 128)
        self.up_conv3 = DoubleConv(64 + 64, 64)

        # Attention Layer
        if join == "attention":
            self.attn_layer = AttentionFusionLayer(
                num_models=num_models, num_channels=num_classes
            )

            # Classify
            self.classify_xi = Classify(64, num_classes)
            self.classify = Classify(num_classes, num_classes)
        if join == "concat":
            self.classify = Classify(64 * 4, num_classes)

    def forward(self, data):
        x = list(data.values())
        feats = []

        for xi in x:
            # Down Convolutions
            xi_1 = self.down_conv1(xi)
            xi_2 = self.down_conv2(xi_1)
            xi_3 = self.down_conv3(xi_2)
            xi_4 = self.down_conv4(xi_3)

            # Up convolutions
            xi = self.tran_conv1(xi_4)
            xi = torch.cat([xi, xi_3], dim=1)
            xi = self.up_conv1(xi)

            xi = self.tran_conv2(xi)
            xi = torch.cat([xi, xi_2], dim=1)
            xi = self.up_conv2(xi)

            xi = self.tran_conv3(xi)
            xi = torch.cat([xi, xi_1], dim=1)
            xi = self.up_conv3(xi)

            if self.join == "concat":
                feats.append(xi)
            elif self.join == "attention":
                xi = self.classify_xi(xi)
                feats.append(xi)

        # Classify
        if self.join == "concat":
            x = torch.cat(feats, dim=1)
        elif self.join == "attention":
            x = self.attn_layer(feats)
        else:
            raise Exception(f"Join type: {self.join} Not Implemented")
        x = self.classify(x)

        return x