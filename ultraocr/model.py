"""OCR model with a CNN backbone + bidirectional GRU + linear decoder."""

import torch
from torch import nn


class OCR(nn.Module):
    """CTC-based OCR model."""

    def __init__(
        self,
        num_of_chars,
        conv_channels=(4, 8, 16, 32),
        kernel_size=3,
        gru_hidden=128,
        gru_num_layers=2,
        gru_bidirectional=True,
    ):
        super().__init__()

        # Build CNN backbone
        in_ch = 3
        layers = []
        for out_ch in conv_channels:
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding="same")
            )
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.ReLU())
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)

        gru_input_size = conv_channels[-1]
        gru_dirs = 2 if gru_bidirectional else 1

        self.biGRUs = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=gru_bidirectional,
        )

        self.decoder = nn.Linear(gru_hidden * gru_dirs, num_of_chars + 1, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean(dim=-2, keepdim=False)
        x = x.permute(0, 2, 1)
        x = self.biGRUs(x)
        x = self.decoder(x[0])
        return x