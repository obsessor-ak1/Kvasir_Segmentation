import torch
from torch import nn
from torch.nn import functional as F


class AttentionGate(nn.Module):
    """The Attention Gate module for Attn UNet."""
    def __init__(self, in_channels, skip_channels, out_channels, num_maps=1):
        super().__init__()
        self.num_maps = num_maps
        self.downsampler = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=skip_channels,
            kernel_size=2,
            stride=2
        )
        self.Wx = nn.Conv2d(
            in_channels=skip_channels, out_channels=out_channels, kernel_size=1
        )
        self.Wg = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.psi = nn.Conv2d(
            in_channels=out_channels, out_channels=num_maps, kernel_size=1, bias=True
        )

    def forward(self, G, X):
        X_down = self.downsampler(X)
        activ1 = F.relu(self.Wx(X_down) + self.Wg(G), inplace=True)
        weights = F.sigmoid(self.psi(activ1))
        weights = F.interpolate(
            weights, size=X.shape[2:], mode="bilinear"
        )
        self.weights = weights
        weights = weights.unsqueeze(2)
        b, _, h, w = X.shape
        X = X.unsqueeze(1)
        attention = weights * X
        attention = attention.reshape(b, -1, h, w)
        return attention


class AttentionUnetDecoderBlock(nn.Module):
    """The Attention UNet Decoder Block for Attn UNet."""
    def __init__(self, gate_channels, skip_channels, attn_channels, out_channels, num_maps=1):
        super().__init__()
        self.attention = AttentionGate(gate_channels, skip_channels, attn_channels, num_maps)
        self.upsampler = nn.ConvTranspose2d(
            in_channels=gate_channels, out_channels=out_channels,
            kernel_size=2, stride=2
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d((out_channels + skip_channels * num_maps), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, G, X):
        attention = self.attention(G, X)
        X = self.upsampler(G)
        X = torch.cat((X, attention), dim=1)
        X = self.conv_block(X)
        return X

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
