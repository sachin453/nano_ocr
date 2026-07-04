"""Backbone providers — timm-based pretrained feature extractors, truncated."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TruncatedMobileNetV3(nn.Module):
    """
    Pretrained MobileNetV3 with only the first *num_blocks* kept.

    Stem (conv_stem → bn1) → blocks[0] → ... → blocks[num_blocks-1]

    When return_features=True, forward() returns a list of intermediate
    feature maps (one per block) for multi-scale fusion.
    """

    def __init__(self, model_name: str, num_blocks: int, pretrained: bool = True,
                 return_features: bool = False):
        super().__init__()

        self.return_features = return_features

        # Load the full model — we only keep the first few layers
        full = timm.create_model(model_name, pretrained=pretrained)

        self.stem = nn.Sequential(full.conv_stem, full.bn1)  # BatchNormAct2d includes activation

        # Slice the desired number of blocks
        blocks = full.blocks[:num_blocks]
        self.blocks = blocks

        # Determine output channels from the last block's last Conv2d
        self._out_channels = self._infer_out_channels(blocks)

        # Collect per-block output channels for multi-scale fusion
        self._block_channels = self._collect_block_channels(blocks)

        # Free the rest of full to save memory
        del full

    def forward(self, x):
        x = self.stem(x)

        if not self.return_features:
            x = self.blocks(x)
            return x

        # Return list of intermediate features (one per block)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def block_channels(self):
        """List of output channels per block (for multi-scale fusion)."""
        return self._block_channels

    @staticmethod
    def _infer_out_channels(blocks):
        """Extract output channels from the last block's final pointwise conv."""
        if len(blocks) == 0:
            raise ValueError("num_blocks must be >= 1 for OCR backbone")
        last_block = blocks[-1]
        convs = [m for m in last_block.modules() if isinstance(m, nn.Conv2d)]
        return convs[-1].out_channels

    @staticmethod
    def _collect_block_channels(blocks):
        """Collect output channels from each block."""
        channels = []
        for block in blocks:
            convs = [m for m in block.modules() if isinstance(m, nn.Conv2d)]
            channels.append(convs[-1].out_channels)
        return channels


class MultiScaleFusion(nn.Module):
    """
    Fuses feature maps from multiple backbone blocks into a single tensor.

    Each block output is projected to *fusion_channels* via 1×1 conv, then
    upsampled to the largest spatial size (from the earliest block), and
    concatenated.  A final 1×1 conv reduces back to *fusion_channels*.
    """

    def __init__(self, block_channels, fusion_channels=48):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(c, fusion_channels, kernel_size=1, bias=False)
            for c in block_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_channels * len(block_channels), fusion_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = fusion_channels

    def forward(self, features):
        """Args: list of (N, C_i, H_i, W_i) tensors from backbone blocks."""
        # Target spatial size = largest (from earliest block, which has most H/W)
        target_h = max(f.shape[2] for f in features)
        target_w = max(f.shape[3] for f in features)

        projected = []
        for proj, feat in zip(self.projections, features):
            x = proj(feat)
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.interpolate(x, size=(target_h, target_w),
                                  mode="bilinear", align_corners=False)
            projected.append(x)

        concat = torch.cat(projected, dim=1)
        return self.fuse(concat)


def create_timm_backbone(model_name: str, num_blocks: int, pretrained: bool = True,
                         return_features: bool = False):
    """
    Build a truncated pretrained backbone that only runs the first *num_blocks*.

    Args:
        model_name: e.g. 'mobilenetv3_small_100.lamb_in1k'
        num_blocks: how many InvertedResidual blocks to keep (1–6)
        pretrained: whether to download ImageNet weights
        return_features: if True, forward() returns a list of per-block features

    Returns:
        backbone: nn.Module taking (N,3,H,W) → (N,C,H_out,W_out) or list
    """
    return TruncatedMobileNetV3(model_name, num_blocks, pretrained, return_features)