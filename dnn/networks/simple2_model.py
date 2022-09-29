import numpy as np

import torch
import torch.nn as nn

class ConvBlockSingle(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(num_inputs, num_outputs, kernel_size, padding='same', padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_outputs),
        )

    def forward(self, inputs):
        return self.model(inputs)

class ConvBlockDouble(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlockSingle(num_features, num_features),
            ConvBlockSingle(num_features, num_features),
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        #x += inputs
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        num_output_conv_features = 1024
        num_input_linear_features = num_output_conv_features * (self.config.rows - 3) * self.config.columns

        self.conv_encoder = nn.Sequential(
            ConvBlockSingle(1, 3, 1),
            ConvBlockSingle(3, 64),

            ConvBlockSingle(64, 128),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(1, 0)),

            ConvBlockSingle(128, 256),

            ConvBlockSingle(256, 512),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(1, 0)),

            ConvBlockSingle(512, 1024),
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
