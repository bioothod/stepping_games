import numpy as np

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        rows = config['rows']
        columns = config['columns']
        num_features = config['num_features']

        num_output_conv_features = 512
        #num_input_linear_features = num_output_conv_features * 4
        num_input_linear_features = num_output_conv_features * (rows - 4) * columns

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding='same', padding_mode='zeros'),
            nn.GELU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, padding='same', padding_mode='zeros'),
            nn.GELU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, padding='same', padding_mode='zeros'),
            nn.GELU(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),

            nn.Conv2d(64, 128, 3, padding='same', padding_mode='zeros'),
            nn.GELU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, padding='same', padding_mode='zeros'),
            nn.GELU(),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(1, 0)),

            nn.Conv2d(256, 512, 3, padding='same', padding_mode='zeros'),
            nn.GELU(),
            nn.BatchNorm2d(512),

        )

        self.linear_encoder = nn.Sequential(
            nn.Flatten(),

            nn.Linear(num_input_linear_features, num_features),
            nn.GELU(),
            nn.BatchNorm1d(num_features),
        )

    def forward(self, inputs):
        conv_features = self.conv_encoder(inputs)
        linear_features = self.linear_encoder(conv_features)

        return linear_features
