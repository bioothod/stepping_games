import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, config, state_features_model):
        super().__init__()

        self.batch_size = config['batch_size']
        self.state_features_model = state_features_model

        hidden_dims = [config['num_features']] + config['hidden_dims'] + [1]

        modules = []

        for i in range(1, len(hidden_dims)):
            input_dim = hidden_dims[i-1]
            output_dim = hidden_dims[i]

            l = nn.Linear(input_dim, output_dim)
            modules.append(l)
            modules.append(nn.ReLU(inplace=True))

        self.values = nn.Sequential(*modules)

    def forward_one(self, inputs):
        with torch.no_grad():
            states_features = self.state_features_model(inputs)

        v = self.values(states_features)
        v = v.squeeze(1)
        return v

    def forward(self, inputs):
        return_values = []

        start_index = 0
        while start_index < len(inputs):
            rest = len(inputs) - (start_index + self.batch_size)
            if rest < 10:
                batch = inputs[start_index:, ...]
            else:
                batch = inputs[start_index:start_index+self.batch_size, ...]

            ret = self.forward_one(batch)
            return_values.append(ret)

            start_index += len(batch)

        return_values = torch.cat(return_values, 0)
        return return_values

class Actor(nn.Module):
    def __init__(self, config, feature_model_creation_func):
        super().__init__()

        self.batch_size = config['batch_size']
        rows = config['rows']
        columns = config['columns']
        self.observation_dtype = np.float32
        self.observation_shape = [1, rows, columns]

        self.train_state_features = True
        self.state_features_model = feature_model_creation_func(config)

        hidden_dims = [config['num_features']] + config['hidden_dims'] + [config['num_actions']]
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

    def create_state(self, player_id, game_state):
        state = game_state
        if player_id == 2:
            state = torch.zeros_like(game_state)
            state[game_state == 2] = 1
            state[game_state == 1] = 2

        return state

    def create_game_from_state(self, player_id, states):
        games = torch.zeros([len(states)] + self.observation_shape, dtype=torch.float32)
        if player_id == 1:
            games[states == 1] = 1
            games[states == 2] = 2
        else:
            games[states == 2] = 1
            games[states == 1] = 2

        return games

    def forward_one(self, inputs):
        if self.train_state_features:
            state_features = self.state_features(inputs)
        else:
            with torch.no_grad():
                state_features = self.state_features(inputs)

        outputs = self.features(state_features)
        return outputs

    def forward(self, player_id, game_states):
        states = self.create_state(player_id, game_states)

        return_logits = []

        start_index = 0
        while start_index < len(states):
            rest = len(states) - (start_index + self.batch_size)
            if rest < 10:
                batch = states[start_index:, ...]
            else:
                batch = states[start_index:start_index+self.batch_size, ...]
            ret = self.forward_one(batch)
            return_logits.append(ret)

            start_index += len(batch)

        return_logits = torch.cat(return_logits, 0)
        return return_logits

    def dist_actions(self, player_id, game_states):
        logits = self.forward(player_id, game_states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()

        log_prob = dist.log_prob(actions)

        is_exploratory = actions != torch.argmax(logits, axis=1)
        return actions, log_prob, is_exploratory

    def select_actions(self, player_id, game_states):
        logits = self.forward(player_id, game_states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions

    def get_predictions(self, player_id, game_states, actions):
        logits = self.forward(player_id, game_states)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropies = dist.entropy()
        return log_prob, entropies

    def greedy_actions(self, player_id, game_states):
        logits = self.forward(player_id, game_states)
        actions = torch.argmax(logits, 1)
        return actions
