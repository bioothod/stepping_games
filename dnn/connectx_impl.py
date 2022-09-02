import joblib

from collections import defaultdict
from easydict import EasyDict

import numpy as np
import torch

@torch.jit.script
def check_reward(game, player_id, num_rows, num_columns, inarow):
    row_player = torch.ones(inarow) * player_id
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
def step_single_game(game, player_id, action, num_rows, num_columns, inarow):
    non_zero = torch.count_nonzero(game[:, :, action])
    if non_zero == num_rows:
        return game, float(-10.), float(1.)

    game[:, num_rows - non_zero - 1, action] = player_id

    reward, done = check_reward(game, player_id, num_rows, num_columns, inarow)

    return game, float(reward), float(done)

class PlayerStat:
    def __init__(self):
        self.timesteps = 0
        self.reward = 0.0
        self.exploration_steps = 0

    def update(self, reward, exploration_step):
        self.timesteps += 1
        self.reward += reward
        self.exploration_steps += exploration_step
        
class GameStat:
    def __init__(self, game_id, player_ids):
        self.game_id = game_id

        self.player_stats = {player_id:PlayerStat() for player_id in player_ids}

    def update(self, player_id, reward, exploration):
        self.player_stats[player_id].update(reward, exploration)

class ConnectX:
    def __init__(self, config: EasyDict, num_games: int):
        self.num_games = num_games
        self.num_actions = config.columns
        self.num_columns = config.columns
        self.num_rows = config.rows
        self.inarow = config.inarow

        self.player_ids = [1, 2]

        self.observation_shape = (1, self.num_rows, self.num_actions)
        self.observation_dtype = torch.float32

        self.total_steps = 0
        self.completed_games = []
        self.new_game_index = 0
        self.reset()

    def create_new_game(self):
        gs = GameStat(self.new_game_index, self.player_ids)
        self.new_game_index += 1
        return gs

    def reset(self):
        self.games = torch.zeros((self.num_games, 1, self.num_rows, self.num_actions), dtype=self.observation_dtype)

        self.current_games = [self.create_new_game() for _ in range(self.num_games)]

        return self.games

    def current_states(self):
        return self.games

    def completed_games_stats(self, num_games):
        rewards = defaultdict(list)
        expl = defaultdict(list)
        time_steps = defaultdict(list)

        for gs in self.completed_games[-num_games:]:
            for player_id, player_stats in gs.player_stats.items():
                rewards[player_id].append(player_stats.reward)
                expl[player_id].append(player_stats.exploration_steps / player_stats.timesteps)
                time_steps[player_id].append(player_stats.timesteps)

        mean_rewards = {player_id:np.mean(rew) for player_id, rew in rewards.items()}
        std_rewards = {player_id:np.std(rew) for player_id, rew in rewards.items()}
        mean_expl = {player_id:np.mean(ex) for player_id, ex in expl.items()}
        std_expl = {player_id:np.std(ex) for player_id, ex in expl.items()}
        mean_timesteps = {player_id:np.mean(ts) for player_id, ts in time_steps.items()}
        std_timesteps = {player_id:np.std(ts) for player_id, ts in time_steps.items()}

        return mean_rewards, std_rewards, mean_expl, std_expl, mean_timesteps, std_timesteps

    def last_game_stats(self):
        return self.completed_games[-1]
    
    def update_game_rewards(self, player_id, rewards, dones, explorations):
        for game_id, (reward, done, exploration) in enumerate(zip(rewards, dones, explorations)):
            gs = self.current_games[game_id]
            gs.update(player_id, reward, exploration)

            if done:
                self.games[game_id, ...] = 0

                for other_player_id in self.player_ids:
                    if other_player_id != player_id:
                        other_reward = reward
                        if reward == 1:
                            other_reward = -1
                        elif reward < 0:
                            other_reward = 1
                            
                        gs.update(other_player_id, other_reward, exploration)

                self.completed_games.append(gs)
                self.current_games[game_id] = self.create_new_game()

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

        self.total_steps += len(self.games)
        return self.games, rewards, dones

    def close(self):
        pass
