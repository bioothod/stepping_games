import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_model import ConvBlock

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        num_output_conv_features = 128
        num_input_linear_features = num_output_conv_features * self.config.rows * self.config.columns

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2),

            nn.Conv2d(2, 128, 4, padding='same', padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        
        self.linear_encoder = nn.Sequential(
            nn.Flatten(),
        
            nn.Linear(num_input_linear_features, self.config.num_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.config.num_features),
        )

    def forward(self, inputs):
        conv_features = self.conv_encoder(inputs)
        linear_features = self.linear_encoder(conv_features)
        
        return linear_features
