import torch
from torch import nn

from seg_modules.architectures.segformer.layers import SegFormerEncoderBlock
from seg_modules.architectures.segformer.model_configs import get_config


class SegFormerEncoder(nn.Module):
    """Implements a SegFormer Encoder module."""

    def __init__(self, input_dim, config):
        super().__init__()
        channels = config["channels"]
        stages_info = config["stages"]
        channels.insert(0, input_dim)
        self.stages = nn.ModuleList([])
        for i in range(1, 5):
            st_conf = stages_info[i - 1]
            self.stages.append(
                SegFormerEncoderBlock(
                    input_dim=channels[i - 1],
                    output_dim=channels[i],
                    stage=i,
                    sr_ratio=st_conf["R"],
                    exp_ratio=st_conf["E"],
                    n_layers=st_conf["L"],
                    num_heads=st_conf["N"],
                )
            )

    def forward(self, X):
        outs = []
        for layer in self.stages:
            X = layer(X)
            outs.append(X)
        return outs


class SegFormerMLPDecoder(nn.Module):
    """The MLP decoder for SegFormer."""

    def __init__(self, config, target_h, target_w, num_classes):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        self.C = config.get("decoder_dim", 256)

        encoder_channels = config["channels"][-4:]

        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, self.C, kernel_size=1),
                    nn.BatchNorm2d(self.C),
                    nn.GELU(),
                )
                for in_ch in encoder_channels
            ]
        )
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(4 * self.C, self.C, kernel_size=1),
            nn.BatchNorm2d(self.C),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(self.C, num_classes, kernel_size=1)

    def forward(self, features):
        assert len(features) == 4, f"Expected 4 stage features, got {len(features)}"
        target_size = features[0].shape[2:]

        unified_features = []
        for i, f in enumerate(features):
            f_proj = self.mlps[i](f)
            if f_proj.shape[2:] != target_size:
                f_proj = nn.functional.interpolate(
                    f_proj, size=target_size, mode="bilinear", align_corners=False
                )
            unified_features.append(f_proj)

        f_concat = torch.cat(unified_features, dim=1)
        f_fused = self.linear_fuse(f_concat)
        logits = self.classifier(f_fused)
        out = nn.functional.interpolate(
            logits,
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        )
        return out


class SegFormer(nn.Module):
    """The full SegFormer model combining SegFormerEncoder and SegFormerMLPDecoder."""

    def __init__(
        self,
        model_name="b0",
        input_dim=3,
        num_classes=1,
        target_h=256,
        target_w=256,
    ):
        super().__init__()
        config = get_config(model_name)
        self.encoder = SegFormerEncoder(input_dim, config)
        self.decoder = SegFormerMLPDecoder(config, target_h, target_w, num_classes)

    def forward(self, X):
        features = self.encoder(X)
        out = self.decoder(features)
        return out


if __name__ == "__main__":
    X = torch.randn(1, 3, 256, 256)
    model = SegFormer(
        model_name="b0", input_dim=3, num_classes=1, target_h=256, target_w=256
    )
    output = model(X)
    print("Output shape:", output.shape)
