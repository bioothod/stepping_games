import numpy as np
import torch
import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dims = [512, 128]
        hidden_dims = [config.num_features] + hidden_dims
        modules = []

        for i in range(1, len(hidden_dims)):
            input_dim = hidden_dims[i-1]
            output_dim = hidden_dims[i]

            l = nn.Linear(input_dim, output_dim)
            modules.append(l)
            modules.append(nn.GELU())
            #modules.append(nn.Dropout(0.3))
        
        self.features = nn.Sequential(*modules)

        self.output_value = nn.Linear(hidden_dims[-1], 1)
        self.output_adv = nn.Linear(hidden_dims[-1], config.num_actions)

    def forward(self, inputs):
        features = self.features(inputs)

        a = self.output_adv(features)
        v = self.output_value(features)
        v = v.expand_as(a)
        
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q
