import torch
from torch import nn


class UNetDownsampleConvBlock(nn.Module):
    """The downsampling block for the UNet Architecture."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        return x


class UNetUpsampleBlock(nn.Module):
    """The upsampling block for the UNet Architecture."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, X, skip_connection):
        X = self.upsample(X)
        concatenated = torch.cat((skip_connection, X), dim=1)
        X = self.conv_block(concatenated)
        return X


class UNet(nn.Module):
    """The UNet Architecture for Medical Image Segmentation."""
    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        self.downsample_convs = nn.ModuleList([
            UNetDownsampleConvBlock(in_channels, 64),
            UNetDownsampleConvBlock(64, 128),
            UNetDownsampleConvBlock(128, 256),
            UNetDownsampleConvBlock(256, 512),
        ])
        self.downsampler = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_downsample = UNetDownsampleConvBlock(512, 1024)
        self.upsamplers = nn.ModuleList([
            UNetUpsampleBlock(1024, 512),
            UNetUpsampleBlock(512, 256),
            UNetUpsampleBlock(256, 128),
            UNetUpsampleBlock(128, 64)
        ])
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, X):
        skip_connections = []
        for downsample_conv in self.downsample_convs:
            X = downsample_conv(X)
            skip_connections.append(X)
            X = self.downsampler(X)
        X = self.final_downsample(X)
        skip_connections = skip_connections[::-1]
        for upsampler, skip_X in zip(self.upsamplers, skip_connections):
            X = upsampler(X, skip_X)
        X = self.classifier(X)
        return X
    