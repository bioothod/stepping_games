from easydict import EasyDict

import torch

@torch.jit.script
def check_reward(game, player, num_rows, num_columns, inarow):
    row_player = torch.ones(inarow) * player
    columns_end = num_columns - (inarow - 1)
    rows_end = num_rows - (inarow - 1)

    for row in torch.arange(0, num_rows, dtype=torch.int64):
        for col in torch.arange(0, columns_end, dtype=torch.int64):
            window = game[:, row, col:col+inarow]
            if torch.all(window == row_player):
                return 1.0

    for col in torch.arange(0, num_columns, dtype=torch.int64):
        for row in torch.arange(0, rows_end, dtype=torch.int64):
            window = game[:, row:row+inarow, col]
            if torch.all(window == row_player):
                return 1.0

    for row in torch.arange(0, rows_end, dtype=torch.int64):
        row_index = torch.arange(row, row+inarow)
        for col in torch.arange(0, columns_end, dtype=torch.int64):
            col_index = torch.arange(col, col+inarow)
            window = game[:, row_index, col_index]
            if torch.all(window == row_player):
                return 1.0

    for row in torch.arange(inarow-1, num_rows, dtype=torch.int64):
        row_index = torch.arange(row, row-inarow, -1)
        for col in torch.arange(0, columns_end, dtype=torch.int64):
            col_index = torch.arange(col, col+inarow)
            window = game[:, row_index, col_index]
            if torch.all(window == row_player):
                return 1.0

    return float(1.0 / float(num_rows * num_columns))

@torch.jit.script
def step(games, player, actions, num_rows, num_columns, inarow):
    rewards = torch.zeros(len(actions), dtype=torch.float32)
    dones = torch.zeros(len(actions), dtype=torch.float32)
    for cont_idx in torch.arange(0, len(games), dtype=torch.int64):
        game = games[cont_idx]
        action = actions[cont_idx]

        non_zero = torch.count_nonzero(game[:, :, action])
        if non_zero == num_rows:
            dones[cont_idx] = 1
            rewards[cont_idx] = -10
            continue

        game[:, num_rows - non_zero - 1, action] = player

        reward = check_reward(game, player, num_rows, num_columns, inarow)
        if reward == 1:
            dones[cont_idx] = 1

        rewards[cont_idx] = reward

    return games, rewards, dones

class ConnectX:
    def __init__(self, config: EasyDict, num_games: int):
        self.num_games = num_games
        self.num_actions = config.columns
        self.num_columns = config.columns
        self.num_rows = config.rows
        self.inarow = config.inarow

        self.observation_shape = (1, self.num_rows, self.num_actions)
        self.observation_dtype = torch.float32

        self.reset()

    def reset(self):
        self.games = torch.zeros((self.num_games, 1, self.num_rows, self.num_actions), dtype=self.observation_dtype)
        return self.games

    def reset_games(self, game_index):
        self.games[game_index, ...] = 0

    def current_games(self):
        return self.games

    def step(self, player, actions):
        self.games, rewards, dones = step(self.games, player, actions, self.num_rows, self.num_columns, self.inarow)
        return self.games, rewards, dones
    
    def close(self):
        pass
