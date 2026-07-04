"""OCR model — timm pretrained backbone + multi-scale fusion + BiGRU + CTC decoder."""

import torch
from torch import nn

from ultraocr.backbones import create_timm_backbone, MultiScaleFusion


class OCR(nn.Module):
    """
    OCR model with a timm pretrained CNN backbone, optional multi-scale feature
    fusion, bidirectional GRU, and CTC-compatible linear decoder.

    The backbone outputs a feature map (N, C, H, W).  We collapse the height
    dimension (mean pool), transpose to (N, T, C), feed through a BiGRU, and
    finally project to character logits via a linear layer.

    When multi_scale is enabled, the backbone returns features from multiple
    blocks, which are fused via 1×1 convs + upsampling before the GRU.

    Args:
        cfg: Config object (from ultraocr.config).
    """

    def __init__(self, cfg):
        super().__init__()

        self.img_h = cfg.model.img_h
        self.img_w = cfg.model.img_w

        # --- Multi-scale config ---
        self.multi_scale = getattr(cfg.model, "multi_scale", False)
        fusion_channels = getattr(cfg.model, "fusion_channels", 48)

        # --- Backbone (timm pretrained, config-driven) ---
        self.backbone = create_timm_backbone(
            model_name=cfg.model.timm_name,
            num_blocks=cfg.model.num_blocks,
            pretrained=cfg.model.pretrained,
            return_features=self.multi_scale,
        )

        # --- Feature fusion (if multi-scale) ---
        if self.multi_scale:
            self.fusion = MultiScaleFusion(
                block_channels=self.backbone.block_channels,
                fusion_channels=fusion_channels,
            )
            backbone_out = self.fusion.out_channels
        else:
            backbone_out = self.backbone.out_channels

        # --- BiGRU ---
        self.biGRU = nn.GRU(
            input_size=backbone_out,
            hidden_size=cfg.model.gru_hidden_size,
            num_layers=cfg.model.gru_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # --- Decoder ---
        gru_out_dim = cfg.model.gru_hidden_size * 2  # bidirectional
        self.decoder = nn.Linear(gru_out_dim, cfg.num_chars + 1, bias=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (N, 3, H, W) input image batch

        Returns:
            logits: (N, T, num_chars + 1) per-frame logits
        """
        if self.multi_scale:
            # Backbone returns list of per-block features
            features = self.backbone(x)
            feats = self.fusion(features)
        else:
            # Backbone returns single feature map
            feats = self.backbone(x)

        # Collapse height with mean pooling → (N, C, W)
        feats = feats.mean(dim=-2, keepdim=False)

        # Transpose to (N, T, C) for GRU
        feats = feats.permute(0, 2, 1)

        # BiGRU
        gru_out, _ = self.biGRU(feats)

        # Linear decode → (N, T, num_chars+1)
        logits = self.decoder(gru_out)

        return logits