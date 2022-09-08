from collections import defaultdict
from easydict import EasyDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def check_reward(games, player_id, num_rows, num_columns, inarow):
    row_player = torch.ones(inarow) * player_id
    columns_end = num_columns - (inarow - 1)
    rows_end = num_rows - (inarow - 1)

    row_player = row_player.to(games.device)
    dones = torch.zeros(len(games), dtype=torch.bool, device=games.device)
    idx = torch.arange(len(games), device=games.device)

    for row in torch.arange(0, num_rows, dtype=torch.int64):
        for col in torch.arange(0, columns_end, dtype=torch.int64):
            window = games[idx, :, row, col:col+inarow]
            win_idx = torch.all(window == row_player, -1)
            win_idx = torch.any(win_idx, 1)
            dones[idx] = torch.logical_or(dones[idx], win_idx)
            idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for col in torch.arange(0, num_columns, dtype=torch.int64):
            for row in torch.arange(0, rows_end, dtype=torch.int64):
                window = games[idx, :, row:row+inarow, col]
                win_idx = torch.all(window == row_player, -1)
                win_idx = torch.any(win_idx, 1)
                dones[idx] = torch.logical_or(dones[idx], win_idx)
                idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for row in torch.arange(0, rows_end, dtype=torch.int64):
            row_index = torch.arange(row, row+inarow)
            for col in torch.arange(0, columns_end, dtype=torch.int64):
                col_index = torch.arange(col, col+inarow)
                window = games[idx][:, :, row_index, col_index]
                win_idx = torch.all(window == row_player, -1)
                win_idx = torch.any(win_idx, 1)
                dones[idx] = torch.logical_or(dones[idx], win_idx)
                idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for row in torch.arange(inarow-1, num_rows, dtype=torch.int64):
            row_index = torch.arange(row, row-inarow, -1)
            for col in torch.arange(0, columns_end, dtype=torch.int64):
                col_index = torch.arange(col, col+inarow)
                window = games[idx][:, :, row_index, col_index]
                win_idx = torch.all(window == row_player, -1)
                win_idx = torch.any(win_idx, 1)
                dones[idx] = torch.logical_or(dones[idx], win_idx)
                idx = idx[torch.logical_not(win_idx)]

    default_reward = float(1.0 / float(num_rows * num_columns))

    rewards = torch.where(dones, 1.0, default_reward)
    dones = torch.where(dones, 1.0, 0.0)
    return rewards, dones

def step_games(games, player_id, actions, num_rows, num_columns, inarow):
    batch_size = actions.shape[0]
    non_zero = torch.count_nonzero(games[torch.arange(batch_size, dtype=torch.int64), :, :, actions], 2).squeeze(1)

    #print(f'actions: {actions.shape}, non_zero: {non_zero.shape}')
    invalid_action_index_batch = non_zero == num_rows
    good_action_index_batch = non_zero < num_rows

    good_actions_index = actions[good_action_index_batch]
    games[good_action_index_batch, :, num_rows - non_zero[good_action_index_batch] - 1, good_actions_index] = player_id

    #print(games[0, 0].type(torch.int32))
    
    rewards, dones = check_reward(games, player_id, num_rows, num_columns, inarow)
    rewards = torch.where(invalid_action_index_batch, float(-10), rewards)
    dones = torch.where(invalid_action_index_batch, float(1), dones)

    return games, rewards, dones

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
        self.device = config.device

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
        self.games = torch.zeros((self.num_games, 1, self.num_rows, self.num_actions), dtype=self.observation_dtype, device=self.device)

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
        reset_game_ids = []
        for game_id, (reward, done, exploration) in enumerate(zip(rewards, dones, explorations)):
            gs = self.current_games[game_id]
            gs.update(player_id, reward, exploration)

            if done:
                reset_game_ids.append(game_id)

                for other_player_id in self.player_ids:
                    if other_player_id != player_id:
                        if reward == 1:
                            other_reward = -1
                        elif reward < 0:
                            other_reward = 1
                            
                        gs.update(other_player_id, other_reward, exploration)

                self.completed_games.append(gs)
                self.current_games[game_id] = self.create_new_game()

        self.games[reset_game_ids, ...] = 0

    def step(self, player_id, actions):
        actions = actions.to(self.device)
        self.games, rewards, dones = step_games(self.games, player_id, actions, self.num_rows, self.num_columns, self.inarow)

        self.total_steps += len(self.games)
        rewards = rewards.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()
        #games = self.games.detach().cpu().numpy()

        return self.games, rewards, dones

    def close(self):
        pass
