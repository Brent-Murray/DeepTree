import torch
import torch.nn as nn


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
    
    
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
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

        # Classify
        self.classify = Classify(64, num_classes)

    def forward(self, data):
        x = list(data.values())
        
        # Down Convolutions
        if len(x) > 1:
            x = torch.cat(x, dim=1)
            x1 = self.down_conv1(x) # down conv 1
        else:
            x1 = self.down_conv1(x[0])  # down conv 1

        
        x2 = self.down_conv2(x1)  # down conv 2
        x3 = self.down_conv3(x2)  # down conv 3
        x4 = self.down_conv4(x3)  # down conv 4

        # Up Convolutions with Transpose Convolutions

        x = self.tran_conv1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv1(x)

        x = self.tran_conv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv2(x)

        x = self.tran_conv3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv3(x)

        # Classify
        x = self.classify(x)

        return x