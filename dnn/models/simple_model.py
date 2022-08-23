import numpy as np

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        self.config = config
        self.num_actions = self.config.columns

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, padding='valid'),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, padding='valid'),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, padding='valid'),
            nn.GELU(),
            nn.Conv2d(128, 256, 4, padding='valid'),
            nn.GELU(),
            nn.Conv2d(256, 512, 4, padding='valid'),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.num_actions + 1),
        )

    def forward(self, inputs):
        enc = self.encoder(inputs)
        return enc
