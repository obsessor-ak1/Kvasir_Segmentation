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
    def __init__(self, in_channels, num_classes=5, depth=5):
        if depth <= 2:
            raise ValueError("depth must be greater than 2")
        super().__init__()
        channels = [64 * (2 ** i) for i in range(depth)]
        
        down_convs = [UNetDownsampleConvBlock(in_channels, channels[0])]
        for i in range(1, depth - 1):
            down_convs.append(UNetDownsampleConvBlock(channels[i-1], channels[i]))
        self.downsample_convs = nn.ModuleList(down_convs)
        
        self.downsampler = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_downsample = UNetDownsampleConvBlock(channels[depth-2], channels[depth-1])
        
        self.upsamplers = nn.ModuleList([
            UNetUpsampleBlock(channels[i] * 2, channels[i])
            for i in reversed(range(depth - 1))
        ])
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)
    
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
    def __init__(self, in_channels, num_classes=5, depth=5):
        if depth <= 2:
            raise ValueError("depth must be greater than 2")
        super().__init__()
        channels = [64 * (2 ** i) for i in range(depth)]
        
        down_convs = [UNetDownsampleConvBlock(in_channels, channels[0])]
        for i in range(1, depth - 1):
            down_convs.append(UNetDownsampleConvBlock(channels[i-1], channels[i]))
        self.downsample_convs = nn.ModuleList(down_convs)
        
        self.downsampler = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_downsample = UNetDownsampleConvBlock(channels[depth-2], channels[depth-1])
        
        self.upsamplers = nn.ModuleList([
            AttentionUnetDecoderBlock(
                gate_channels=channels[i+1],
                skip_channels=channels[i],
                attn_channels=(channels[i-1] if i > 0 else channels[0] // 2),
                out_channels=channels[i],
                num_maps=num_classes
            )
            for i in reversed(range(depth - 1))
        ])
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

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
    def __init__(self, in_channels, num_classes=1, inf_avg=True, depth=5):
        if depth <= 2:
            raise ValueError("depth must be greater than 2")
        super().__init__()
        self._depth = depth
        channels = [64 * (2 ** i) for i in range(depth)]
        
        down_convs = [UNetDownsampleConvBlock(in_channels, channels[0])]
        for i in range(1, depth - 1):
            down_convs.append(UNetDownsampleConvBlock(channels[i-1], channels[i]))
        self.downsample_convs = nn.ModuleList(down_convs)
        
        self.downsampler = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_downsample = UNetDownsampleConvBlock(channels[depth-2], channels[depth-1])
        
        self.skip_nodes = nn.ModuleList([
            nn.ModuleList([
                UnetPlusPlusSkipNode(level=i, skip_id=j) for j in range(1, depth-i-1)
            ])
            for i in range(depth-2)
        ])
        self.upsample_convs = nn.ModuleList([
            UNetUpsampleBlock(channels[depth - j - 1] * (j + 1), channels[depth - j - 1])
            for j in range(1, depth)
        ])
        self.classifiers = nn.ModuleList([
            nn.Conv2d(channels[0], num_classes, kernel_size=1)
            for _ in range(depth - 1)
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