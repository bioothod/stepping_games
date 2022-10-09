import numpy as np

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_actions = self.config.columns

        num_output_conv_features = 128
        num_input_linear_features = num_output_conv_features * (self.config.rows) * (self.config.columns)

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 3, 1, padding='same', padding_mode='zeros'),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            
            nn.Conv2d(3, 128, 4, padding='same', padding_mode='zeros'),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 4, padding='same', padding_mode='zeros'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        class Debug(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                print(x.shape)
                return x
        
        self.linear_encoder = nn.Sequential(
            nn.Flatten(),

            nn.Linear(num_input_linear_features, self.config.num_features),
            nn.ReLU(),
            nn.BatchNorm1d(self.config.num_features),
        )

    def forward(self, inputs):
        conv_features = self.conv_encoder(inputs)
        linear_features = self.linear_encoder(conv_features)
        
        return linear_features
