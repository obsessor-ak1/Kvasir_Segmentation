from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F


class MixAttentionBlock(nn.Module):
    """The Mix Transformer block for SegFormer Encoder."""

    def __init__(self, input_dim, sr_ratio, head_count):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.norm1 = nn.LayerNorm(input_dim)
        self.reshaper = nn.Conv2d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=sr_ratio,
            stride=sr_ratio,
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=head_count, batch_first=True
        )
        self.attn_weights = None

    def forward(self, X):
        H, W = X.shape[2:]
        X = rearrange(X, "b c h w -> b h w c")
        X = self.norm1(X)
        X = rearrange(X, "b h w c -> b c h w")
        Q = X
        X = self.reshaper(X)
        X = rearrange(X, "b c h w -> b (h w) c")
        X = self.norm2(X)
        Q = rearrange(Q, "b c h w -> b (h w) c")
        attn_output, attn_weights = self.attention(Q, X, X)
        self.attn_weights = attn_weights
        final_output = attn_output + Q
        final_output = rearrange(final_output, "b (h w) c -> b c h w", h=H, w=W)
        return final_output


class MixFFN(nn.Module):
    """The FFN component of Mix Transformer for Segformer."""

    def __init__(self, input_dim, exp_ratio):
        super().__init__()
        hidden_dim = int(exp_ratio * input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.depth_wise_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, X):
        old = X
        X = rearrange(X, "b c h w -> b h w c")
        X = self.norm(X)
        X = self.mlp1(X)
        X = rearrange(X, "b h w c -> b c h w")
        X = self.depth_wise_conv(X)
        X = self.gelu(X)
        X = rearrange(X, "b c h w -> b h w c")
        X = self.mlp2(X)
        X = rearrange(X, "b h w c -> b c h w")
        final = X + old
        return final


class MixTransformerBlock(nn.Module):
    """The SegFormer Encoder transformer block."""

    def __init__(self, input_dim, sr_ratio, exp_ratio, head_count):
        super().__init__()
        self.attention = MixAttentionBlock(input_dim, sr_ratio, head_count)
        self.mlp = MixFFN(input_dim, exp_ratio)

    def forward(self, X):
        X = self.attention(X)
        X = self.mlp(X)
        return X


class PatchOverlapMerger(nn.Module):
    """Implements patch overlap merging."""

    def __init__(self, stage, input_dim, output_dim):
        super().__init__()
        if stage == 1:
            K, S, P = 7, 4, 3
        elif 2 <= stage <= 4:
            K, S, P = 3, 2, 1
        else:
            raise ValueError("stage must be between 1 and 4")
        self.patcher = nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=K,
            padding=P,
            stride=S,
        )

    def forward(self, X):
        return self.patcher(X)


class SegFormerEncoderBlock(nn.Module):
    """Represents a single SegFormer encoder block."""

    def __init__(
        self, input_dim, output_dim, stage, sr_ratio, exp_ratio, n_layers, num_heads
    ):
        super().__init__()
        self.patch_mergerer = PatchOverlapMerger(stage, input_dim, output_dim)
        self.layers = nn.Sequential(
            *[
                MixTransformerBlock(
                    input_dim=output_dim,
                    sr_ratio=sr_ratio,
                    exp_ratio=exp_ratio,
                    head_count=num_heads,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, X):
        X = self.patch_mergerer(X)
        features = self.layers(X)
        return features


if __name__ == "__main__":
    X = torch.randn(1, 16, 16, 16)
    layer = MixTransformerBlock(16, 4, 2, 2)
    opt = layer(X)
    print(opt.shape)
    print(layer.attention.attn_weights.shape)
