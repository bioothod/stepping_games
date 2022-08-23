import numpy as np
import torch
import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, config):
        self.input_layer = nn.Linear(config.num_features)

        output_features = 64
        
        self.features = nn.Sequential(
            nn.Linear(config.num_features, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, output_features),
            nn.GELU(),
        )

        self.output_value = nn.Linear(output_features, 1)
        self.output_adv = nn.Linear(output_features, config.num_actions)

    def forward(self, inputs):
        features = self.features(inputs)

        a = self.output_adv(features)
        v = self.output_value(features)
        v = v.expand_as(a)
        
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q
