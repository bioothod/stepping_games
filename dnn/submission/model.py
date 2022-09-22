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
        num_input_linear_features = num_output_conv_features * (rows - 3) * columns

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

            nn.Linear(num_input_linear_features, num_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features),
        )

    def forward(self, inputs):
        conv_features = self.conv_encoder(inputs)
        linear_features = self.linear_encoder(conv_features)

        return linear_features

class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.state_features_model = Model(config)

        num_features = config['num_features']
        num_actions = config['num_actions']
        self.rows = config['rows']
        self.columns = config['columns']
        self.player_ids = config['player_ids']

        self.observation_dtype = np.float32
        self.observation_shape = [self.rows, self.columns]

        hidden_dims = [num_features] + config['hidden_dims'] + [num_actions]
        modules = []

        for i in range(1, len(hidden_dims)):
            input_dim = hidden_dims[i-1]
            output_dim = hidden_dims[i]

            l = nn.Linear(input_dim, output_dim)
            modules.append(l)
            modules.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*modules)

    def state_features(self, inputs):
        state_features = self.state_features_model(inputs)
        return state_features

    def create_state(self, obs):
        orig_state = np.asarray(obs['board'], dtype=self.observation_dtype).reshape(self.observation_shape)
        player_id = obs['mark']

        state = torch.zeros((1 + len(self.player_ids), self.rows, self.columns), dtype=torch.float32)
        state[0, ...] = player_id

        for idx, pid in enumerate(self.player_ids):
            player_idx = orig_state == pid
            state[idx + 1, player_idx] = 1

        return state

    def dist_actions(self, inputs):
        state_features = self.state_features(inputs)
        logits = self.features(state_features)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()

        log_prob = dist.log_prob(actions)

        is_exploratory = actions != torch.argmax(logits, axis=1)
        return actions, log_prob, is_exploratory

    def greedy_actions(self, states):
        state_features = self.state_features(states)
        logits = self.features(state_features)

        actions = torch.argmax(logits, 1)
        return actions

    def forward(self, observation):
        state = self.create_state(observation)

        states = state.unsqueeze(0)
        actions = self.greedy_actions(states)

        action = actions.squeeze(0).detach().cpu().numpy()
        return int(action)

default_config = {
    'checkpoint_path': 'submission.ckpt',
    'rows': 6,
    'columns': 7,
    'inarow': 4,
    'num_actions': 7,
    'num_features': 512,
    'hidden_dims': [128],
    'player_ids': [1, 2],
}
