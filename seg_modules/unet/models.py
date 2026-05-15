from torch import nn

from seg_modules.unet.layers import (
    AttentionUnetDecoderBlock,
    UNetUpsampleBlock,
    UNetDownsampleConvBlock
)


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
    

class AttentionUNet(nn.Module):
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
            AttentionUnetDecoderBlock(
                1024, 512, 64, 512, num_maps=num_classes),
            AttentionUnetDecoderBlock(
                512, 256, 32, 256, num_maps=num_classes),
            AttentionUnetDecoderBlock(
                256, 128, 16, 128, num_maps=num_classes),
            AttentionUnetDecoderBlock(
                128, 64, 18, 64, num_maps=num_classes)
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