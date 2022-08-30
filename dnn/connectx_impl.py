from easydict import EasyDict

import torch

class ConnectX:
    def __init__(self, config: EasyDict, num_games: int):
        self.num_games = num_games
        self.num_actions = config.columns
        self.num_columns = config.columns
        self.num_rows = config.rows
        self.inarow = config.inarow

        self.columns_end = self.num_columns - (self.inarow - 1)
        self.row_end = self.num_rows - (self.inarow - 1)

        self.default_reward = 1.0 / (self.num_rows * self.num_columns)

        self.observation_shape = (1, self.num_rows, self.num_actions)
        self.observation_dtype = torch.float32

        self.reset()

    def reset(self):
        self.games = torch.zeros((self.num_games, 1, self.num_rows, self.num_actions), dtype=self.observation_dtype)
        return self.games

    def check_reward(self, game, player):
        row_player = torch.ones(self.inarow) * player

        for row in range(self.num_rows):
            for col in range(self.columns_end):
                window = game[:, row, col:col+self.inarow]
                if torch.all(window == row_player):
                    return 1

        for col in range(self.num_columns):
            for row in range(self.row_end):
                window = game[:, row:row+self.inarow, col]
                if torch.all(window == row_player):
                    return 1

        for row in range(self.row_end):
            row_index = torch.arange(row, row+self.inarow)
            for col in range(self.columns_end):
                col_index = torch.arange(col, col+self.inarow)
                window = game[:, row_index, col_index]
                if torch.all(window == row_player):
                    return 1

        for row in range(self.inarow-1, self.num_rows):
            row_index = torch.arange(row, row-self.inarow, -1)
            for col in range(self.columns_end):
                col_index = torch.arange(col, col+self.inarow)
                window = game[:, row_index, col_index]
                if torch.all(window == row_player):
                    return 1

        return self.default_reward
    
    def step(self, player, game_index, actions):
        if actions.max() >= self.num_actions or actions.min() < 0:
            raise ValueError(f'min_action: {actions.min()}, max_action: {actions.max()}: actions must be in [0, {self.num_actions})')

        rewards = torch.zeros(len(actions), dtype=torch.float32)
        dones = torch.zeros(len(actions), dtype=torch.float32)
        for cont_idx, game_idx in enumerate(game_index):
            game = self.games[game_idx]
            action = actions[cont_idx]

            non_zero = torch.count_nonzero(game[:, :, action])
            if non_zero == self.num_rows:
                dones[cont_idx] = 1
                rewards[cont_idx] = -10
                continue

            game[:, self.num_rows - non_zero - 1, action] = player

            reward = self.check_reward(game, player)
            if reward == 1:
                dones[cont_idx] = 1

            rewards[cont_idx] = reward

        return self.games, rewards, dones

    def close(self):
        pass
