import numpy as np

import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.conv0 = nn.Conv2d(num_features, num_features, 4, padding='same')
        self.activation0 = nn.LeakyReLU()
        self.b0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features, 4, padding='same')
        self.activation1 = nn.LeakyReLU()
        self.b1 = nn.BatchNorm2d(num_features)

    def forward(self, inputs):
        x = self.conv0(inputs)
        x = self.b0(x)
        x = self.activation0(x)

        x = self.conv1(x)
        x = self.b1(x)
        x += inputs
        x = self.activation1(x)
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        rows = config['rows']
        columns = config['columns']
        num_features = config['num_features']

        num_output_conv_features = 32
        num_input_linear_features = num_output_conv_features * rows * columns

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, num_output_conv_features, 4, padding='same', padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_output_conv_features),

            Residual(num_output_conv_features),
            Residual(num_output_conv_features),
            Residual(num_output_conv_features),
            Residual(num_output_conv_features),
            Residual(num_output_conv_features),
            Residual(num_output_conv_features),
        )

        self.linear_encoder = nn.Sequential(
            nn.Flatten(),
        
            nn.Linear(num_input_linear_features, num_features),
            nn.LeakyReLU(),
            nn.LayerNorm(num_features),
        )

    def forward(self, inputs):
        conv_features = self.conv_encoder(inputs)
        linear_features = self.linear_encoder(conv_features)

        return linear_features
