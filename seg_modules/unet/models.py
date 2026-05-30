import torch
from torch import nn

from seg_modules.unet.layers import (
    AttentionUnetDecoderBlock,
    UNetUpsampleBlock,
    UNetDownsampleConvBlock,
    UnetPlusPlusSkipNode
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
                1024, 512, 256, 512, num_maps=num_classes),
            AttentionUnetDecoderBlock(
                512, 256, 128, 256, num_maps=num_classes),
            AttentionUnetDecoderBlock(
                256, 128, 64, 128, num_maps=num_classes),
            AttentionUnetDecoderBlock(
                128, 64, 32, 64, num_maps=num_classes)
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


class UNetPlusPlus(nn.Module):
    """UNet++ implementation for kvasir segmentation."""
    def __init__(self, in_channels, num_classes=1, inf_avg=True):
        super().__init__()
        depth = 5
        self._depth = depth
        self.downsample_convs = nn.ModuleList([
            UNetDownsampleConvBlock(in_channels, 64),
            UNetDownsampleConvBlock(64, 128),
            UNetDownsampleConvBlock(128, 256),
            UNetDownsampleConvBlock(256, 512)
        ])
        self.downsampler = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_downsample = UNetDownsampleConvBlock(512, 1024)
        self.skip_nodes = nn.ModuleList([
            nn.ModuleList([
                UnetPlusPlusSkipNode(level=i, skip_id=j) for j in range(1, depth-i-1)
            ])
            for i in range(depth-2)
        ])
        self.upsample_convs = nn.ModuleList([
            UNetUpsampleBlock(512 * 2, 512),
            UNetUpsampleBlock(256 * 3, 256),
            UNetUpsampleBlock(128 * 4, 128),
            UNetUpsampleBlock(64 * 5, 64)
        ])
        self.classifiers = nn.ModuleList([
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Conv2d(64, num_classes, kernel_size=1),
        ])
        self._inference_branch_count = depth
        self.inf_avg = inf_avg
    
    @property
    def inference_branch_count(self):
        return self._inference_branch_count
    
    @inference_branch_count.setter
    def inference_branch_count(self, count):
        self._inference_branch_count = min(count, self._depth - 1)

    def forward(self, X):
        traversal_length = self._depth
        if not self.training:
            traversal_length = self.inference_branch_count
        
        immediate_results = [[] for _ in range(traversal_length)]
        for i in range(traversal_length):
            if i < self._depth-1:
                X = self.downsample_convs[i](X)
            else:
                X = self.final_downsample(X)
                # And things will stop after here
            immediate_results[i].append(X)
            X = self.downsampler(X)
        
        for j in range(1, traversal_length):
            for i in range(traversal_length-j):
                skip_maps = torch.cat(immediate_results[i], dim=1)
                semantic_map = immediate_results[i+1][j-1]
                if i == traversal_length-j-1:
                    block = self.upsample_convs[j-1]
                else:
                    block = self.skip_nodes[i][j-1]
                out = block(semantic_map, skip_maps)
                immediate_results[i].append(out)

        logits = []
        for i, output in enumerate(immediate_results[0][1:]):
            logit = self.classifiers[i](output)
            logits.append(logit)

        if not self.training:
            if self.inf_avg:
                final_logits = torch.cat(logits, dim=1).mean(dim=1, keepdim=True)
            else:
                final_logits = logits[-1]
        else:
            final_logits = logits
        return final_logits