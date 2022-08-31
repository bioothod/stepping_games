import joblib

from easydict import EasyDict

import numpy as np
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
                return 1.0, 1.0

    for col in torch.arange(0, num_columns, dtype=torch.int64):
        for row in torch.arange(0, rows_end, dtype=torch.int64):
            window = game[:, row:row+inarow, col]
            if torch.all(window == row_player):
                return 1.0, 1.0

    for row in torch.arange(0, rows_end, dtype=torch.int64):
        row_index = torch.arange(row, row+inarow)
        for col in torch.arange(0, columns_end, dtype=torch.int64):
            col_index = torch.arange(col, col+inarow)
            window = game[:, row_index, col_index]
            if torch.all(window == row_player):
                return 1.0, 1.0

    for row in torch.arange(inarow-1, num_rows, dtype=torch.int64):
        row_index = torch.arange(row, row-inarow, -1)
        for col in torch.arange(0, columns_end, dtype=torch.int64):
            col_index = torch.arange(col, col+inarow)
            window = game[:, row_index, col_index]
            if torch.all(window == row_player):
                return 1.0, 1.0

    return float(1.0 / float(num_rows * num_columns)), 0.

@torch.jit.script
def step_single_game(game, player, action, num_rows, num_columns, inarow):
    non_zero = torch.count_nonzero(game[:, :, action])
    if non_zero == num_rows:
        return game, float(-10.), float(1.)

    game[:, num_rows - non_zero - 1, action] = player

    reward, done = check_reward(game, player, num_rows, num_columns, inarow)

    return game, float(reward), float(done)

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
        player = torch.FloatTensor([player])[0]
        jobs = []
        for cont_idx in range(0, len(self.games)):
            game = self.games[cont_idx]
            action = actions[cont_idx]

            job = joblib.delayed(step_single_game)(game, player, action, self.num_rows, self.num_columns, self.inarow)
            jobs.append(job)

        with joblib.parallel_backend('threading', n_jobs=16):
            results = joblib.Parallel(require='sharedmem')(jobs)

            rewards = np.zeros(len(actions), dtype=np.float32)
            dones = np.zeros(len(actions), dtype=np.float32)
            
            for idx, res_tuple in enumerate(results):
                game, reward, done = res_tuple
                
                self.games[idx, ...] = game
                rewards[idx] = reward
                dones[idx] = done
            
        return self.games, rewards, dones
    
    def close(self):
        pass
