import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size=3, activation=F.gelu):
        super().__init__()

        self.activation = activation
        self.conv = nn.Conv2d(num_inputs, num_outputs, kernel_size, padding='same', padding_mode='zeros')
        self.bn = nn.BatchNorm2d(num_outputs)
        self.shortcut = None
        if num_outputs != num_inputs:
            self.shortcut = nn.Conv2d(num_inputs, num_outputs, 1)

    def forward(self, inputs):
        x = self.conv(inputs)
        shortcut = inputs
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)

        x += shortcut
        x = self.activation(x)
        x = self.bn(x)
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 2, 1),
            #nn.BatchNorm2d(3),
            ConvBlock(2, 32, 4),
            ConvBlock(32, 64, 4),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, 4),
            ConvBlock(128, 256, 4),
            nn.MaxPool2d(2),
            ConvBlock(256, 512, 4),
        )

    def forward(self, inputs):
        conv_features = self.conv_encoder(inputs)
        batch_size, channels = conv_features.shape[:2]
        features = torch.reshape(conv_features, [batch_size, -1])
        return features
