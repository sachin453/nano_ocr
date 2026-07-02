"""Focal CTC Loss with config support."""

import torch
from torch import nn


class FocalCTCLoss(nn.Module):
    """Focal CTC Loss — weights hard examples more heavily."""

    def __init__(self, blank=0, alpha=1.0, gamma=2.0, cfg=None):
        super().__init__()

        if cfg is not None:
            blank = cfg.loss.blank_idx
            alpha = cfg.loss.focal_alpha
            gamma = cfg.loss.focal_gamma

        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="none")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        # Clamping prevents negative values from triggering math explosion.
        loss = torch.clamp(loss, min=0.0)

        p = torch.exp(-loss)
        focal_loss = self.alpha * ((1 - p) ** self.gamma) * loss

        return focal_loss.mean()