import numpy as np

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        num_output_conv_features = 512
        num_input_linear_features = num_output_conv_features * (self.config.rows - 3) * self.config.columns

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding='same', padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 32, 3, padding='same', padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, padding='same', padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(1, 0)),

            nn.Conv2d(64, 128, 3, padding='same', padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, padding='same', padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(1, 0)),

            nn.Conv2d(256, 512, 3, padding='same', padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
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
