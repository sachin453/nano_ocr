"""OCR model with a CNN backbone + bidirectional GRU + linear decoder."""

import torch
from torch import nn


class OCR(nn.Module):
    def __init__(self,num_of_chars):  
        super().__init__()

        self.img_w = 512
        self.img_h = 64
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,stride=1,padding='same'),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=4,out_channels=16,kernel_size=3,stride=1,padding='same'),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3,stride=1,padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.biGRUs = nn.GRU(input_size=64,hidden_size=128,num_layers=2,batch_first=True,bidirectional=True)
        self.decoder = nn.Linear(256, num_of_chars + 1, bias=True)

    def forward(self, x):
        x = self.backbone(x) ### out = N x 64 x 8 x 64
        x = x.mean(dim=-2, keepdim=False) ### out = N x 64 x 64
        x = x.permute(0, 2, 1)
        x = self.biGRUs(x)
        x = self.decoder(x[0])
        return x