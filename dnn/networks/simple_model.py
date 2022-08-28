import numpy as np

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_actions = self.config.columns

        num_output_conv_features = 1
        num_input_linear_features = num_output_conv_features * self.config.rows * self.config.columns
        num_linear_features = num_output_conv_features * self.config.rows * self.config.columns * 10
        
        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(1),
            
            nn.Conv2d(1, 16, 4, padding='same', padding_mode='zeros'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, num_output_conv_features, 4, padding='same', padding_mode='zeros'),
            nn.ReLU(),
            nn.BatchNorm2d(num_output_conv_features),
        )

        class Debug(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                print(x.shape)
                return x
        
        self.linear_encoder = nn.Sequential(
            nn.BatchNorm1d(num_input_linear_features),

            nn.Linear(num_input_linear_features, num_linear_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_linear_features),
            
            nn.Linear(num_linear_features, num_linear_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_linear_features),

            nn.Linear(num_linear_features, num_linear_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_linear_features),

            nn.Linear(num_linear_features, num_linear_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_linear_features),

            nn.Linear(num_linear_features, num_linear_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_linear_features),

            nn.Linear(num_linear_features, num_linear_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_linear_features),

            nn.Linear(num_linear_features, num_linear_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_linear_features),

            nn.Linear(num_linear_features, self.config.num_features),
            nn.ReLU(),
        )

    def forward(self, inputs):
        #conv_features = self.conv_encoder(inputs)
        conv_features = inputs
        batch_size, channels = conv_features.shape[:2]
        features = torch.reshape(conv_features, [batch_size, -1])

        linear_features = self.linear_encoder(features)
        
        return linear_features
