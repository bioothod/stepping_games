import joblib

from collections import defaultdict
from easydict import EasyDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def check_reward_single_game(game, player_id, num_rows, num_columns, inarow):
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
def check_reward(games, player_id, num_rows, num_columns, inarow):
    row_player = torch.ones(inarow) * player_id
    columns_end = num_columns - (inarow - 1)
    rows_end = num_rows - (inarow - 1)

    row_player = row_player.to(games.device)
    dones = torch.zeros(len(games), dtype=torch.bool, device=games.device)
    idx = torch.arange(len(games), device=games.device)

    for row in torch.arange(0, num_rows, dtype=torch.int64):
        for col in torch.arange(0, columns_end, dtype=torch.int64):
            window = games[idx][:, :, row, col:col+inarow]
            win_idx = torch.all(window == row_player, -1)
            win_idx = torch.any(win_idx, 1)
            dones[idx] = torch.logical_or(dones[idx], win_idx)
            idx = idx[torch.logical_not(win_idx)]

    if len(idx) > 0:
        for col in torch.arange(0, num_columns, dtype=torch.int64):
            for row in torch.arange(0, rows_end, dtype=torch.int64):
                window = games[idx][:, :, row:row+inarow, col]
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
    #default_reward = 0.0

    rewards = torch.where(dones, 1.0, default_reward)

    return rewards, dones

@torch.jit.script
def step_single_game(game, player_id, action, num_rows, num_columns, inarow):
    non_zero = torch.count_nonzero(game[:, :, action])
    if non_zero == num_rows:
        return game, float(-10.), float(1.)

    game[:, num_rows - non_zero - 1, action] = player_id

    reward, done = check_reward_single_game(game, player_id, num_rows, num_columns, inarow)

    return game, float(reward), float(done)

@torch.jit.script
def step_games(games, player_id, actions, num_rows, num_columns, inarow):
    player_id = torch.tensor(player_id, dtype=torch.float32)

    num_games = len(games)
    non_zero = torch.count_nonzero(games[torch.arange(num_games, dtype=torch.int64), :, :, actions], 2).squeeze(1)

    invalid_action_index_batch = non_zero == num_rows
    good_action_index_batch = non_zero < num_rows

    good_actions_index = actions[good_action_index_batch]
    games[good_action_index_batch, :, num_rows - non_zero[good_action_index_batch] - 1, good_actions_index] = player_id

    rewards, dones = check_reward(games, player_id, num_rows, num_columns, inarow)
    rewards[invalid_action_index_batch] = torch.tensor(-10., dtype=torch.float32)
    dones[invalid_action_index_batch] = True

    return games, rewards, dones


@torch.jit.script
def calculate_rewards(values, rewards, next_value, max_episode_len, gamma, tau):
    episode_len = len(rewards)

    next_value = next_value.unsqueeze(0)
    episode_rewards = torch.cat([rewards, next_value])
    discounts = torch.logspace(0, max_episode_len+1, steps=max_episode_len+1, base=gamma)
    discounts = discounts[:len(episode_rewards)]

    episode_returns = []
    for t in range(episode_len):
        ret = torch.sum(discounts[:len(discounts)-t] * episode_rewards[t:])
        episode_returns.append(ret)

    episode_values = torch.cat([values, next_value], 0)

    episode_values = episode_values.flatten()

    tau_discounts = torch.logspace(0, max_episode_len+1, steps=max_episode_len+1, base=gamma*tau)
    tau_discounts = tau_discounts[:episode_len]
    deltas = episode_rewards[:-1] + gamma * episode_values[1:] - episode_values[:-1]
    gaes = []
    for t in range(episode_len):
        ret = torch.sum(tau_discounts[:len(tau_discounts)-t] * deltas[t:])
        gaes.append(ret)

    return episode_returns, gaes


class ConnectX:
    def __init__(self, config: EasyDict, critic, summary_writer, summary_prefix: str, global_step: torch.Tensor):
        self.summary_writer = summary_writer
        self.summary_prefix = summary_prefix
        self.global_step = global_step

        self.num_games = config.num_training_games
        self.num_actions = config.columns
        self.num_columns = config.columns
        self.num_rows = config.rows
        self.inarow = config.inarow
        self.device = config.device
        self.max_episode_len = config.max_episode_len
        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size

        self.player_ids = config.player_ids

        self.total_games_completed = 0

        self.critic = critic

        self.observation_shape = (1, self.num_rows, self.num_actions)
        self.observation_dtype = torch.float32

        self.reset()

    def reset(self):
        self.games = torch.zeros((self.num_games,) + self.observation_shape, dtype=self.observation_dtype, device='cpu')

        self.states = torch.zeros((self.num_games, self.max_episode_len) + self.observation_shape, dtype=torch.float32, device='cpu')
        self.rewards = torch.zeros((self.num_games, self.max_episode_len), dtype=torch.float32, device='cpu')
        self.actions = torch.zeros((self.num_games, self.max_episode_len), device='cpu').long()
        self.player_id = torch.zeros((self.num_games, self.max_episode_len), device='cpu').long()
        self.log_probs = torch.zeros_like(self.rewards, device='cpu')
        self.explorations = torch.zeros((self.num_games, self.max_episode_len), dtype=torch.bool, device='cpu')
        self.next_values = torch.zeros_like(self.rewards, device='cpu')
        self.dones = torch.zeros(self.num_games, dtype=torch.bool, device='cpu')
        self.episode_len = torch.zeros(self.num_games, device='cpu').long()

    def set_index_state(self, index, states):
        self.games[index, ...] = states.to(self.games.device).detach().clone()
        self.dones = torch.ones_like(self.dones, dtype=torch.bool)
        self.dones[index] = False

    def make_opposite(self, state):
        state_opposite = torch.zeros_like(state)
        state_opposite[state == 1] = 2
        state_opposite[state == 2] = 1
        return state_opposite

    def make_states(self, player_id, games):
        states = games

        if player_id == 2:
            states = self.make_opposite(states)

        return states

    def current_states(self, player_id):
        game_index = self.running_index()
        games = self.games[game_index]

        states = self.make_states(player_id, games)
        return game_index, states

    def running_index(self):
        index = torch.arange(len(self.games))
        return index[torch.logical_not(self.dones)]

    def completed_index(self):
        index = torch.arange(len(self.games))
        completed_index = index[self.dones]
        return completed_index

    def completed_games_stats(self, num_games=None):
        game_index = self.completed_index()
        if num_games:
            game_index = game_index[:num_games]

        rewards = []
        explorations = []
        for game_id in game_index:
            game_rewards = []
            game_explorations = []
            for player_id in self.player_ids:
                player_index = self.player_id[game_id] == player_id
                reward = self.rewards[game_id, player_index].sum(0)
                game_rewards.append(reward)

                exploration = self.explorations[game_id, player_index].float().mean()
                game_explorations.append(exploration)

            rewards.append(game_rewards)
            explorations.append(game_explorations)

        rewards = torch.tensor(rewards)
        explorations = torch.tensor(explorations)


        std_rewards, mean_rewards = torch.std_mean(rewards, 0)
        std_explorations, mean_explorations = torch.std_mean(explorations, 0)
        std_timesteps, mean_timesteps = torch.std_mean(self.episode_len[game_index].float())

        mean_rewards = mean_rewards.cpu().numpy()
        std_rewards = std_rewards.cpu().numpy()
        mean_explorations = mean_explorations.cpu().numpy()
        std_explorations = std_explorations.cpu().numpy()

        return mean_rewards, std_rewards, mean_explorations, std_explorations, float(mean_timesteps), float(std_timesteps)

    def last_game_stats(self):
        game_id = self.completed_index()[-1]
        rewards = []

        for player_id in self.player_ids:
            player_index = self.player_id[game_id] == player_id
            reward = self.rewards[game_id, player_index].sum(0)
            rewards.append(reward)

        rewards = torch.tensor(rewards)
        episode_len = self.episode_len[game_id]

        return int(episode_len), rewards

    def completed_games_and_states(self):
        completed_index = self.completed_index()

        episode_len = self.episode_len[completed_index]
        total_states = torch.sum(episode_len)
        return len(completed_index), int(total_states)

    def update_game_rewards(self, player_id, game_index, states, actions, log_probs, rewards, dones, explorations, next_values):
        episode_index = self.episode_len[game_index]

        self.player_id[game_index, episode_index] = player_id
        self.states[game_index, episode_index] = states.detach().clone().to(self.states.device)
        self.rewards[game_index, episode_index] = rewards.detach().clone().to(self.states.device)
        self.actions[game_index, episode_index] = actions.detach().clone().to(self.states.device)
        self.log_probs[game_index, episode_index] = log_probs.detach().clone().to(self.states.device)
        self.explorations[game_index, episode_index] = explorations.detach().clone().to(self.states.device)
        self.next_values[game_index, episode_index] = next_values.detach().clone().to(self.states.device)
        self.dones[game_index] = dones.detach().clone().to(self.states.device)

        cur_win_index = rewards == 1
        cur_win_game_index = game_index[cur_win_index]
        cur_win_episode_index = episode_index[cur_win_index]
        self.rewards[cur_win_game_index, cur_win_episode_index-1] = -1

        self.episode_len[game_index] += 1
        completed_episodes_index = self.episode_len[game_index] == self.max_episode_len
        completed_game_index = game_index[completed_episodes_index]
        self.dones[completed_game_index] = True

        self.total_games_completed += int(dones.sum().cpu().numpy())

    def dump(self):
        game_index = self.completed_index()

        ret_states = []
        ret_actions = []
        ret_log_probs = []
        ret_gaes = []
        ret_returns = []

        for game_id in game_index:
            episode_len = self.episode_len[game_id]

            states = self.states[game_id, :episode_len, ...]
            ret_states.append(states)
        ret_states = torch.cat(ret_states, 0).to(self.device)

        if len(ret_states) == 0:
            raise ValueError(f'dump: game_index: {len(game_index), ret_states: {len(ret_states)}}: zero-length states array')

        with torch.no_grad():
            ret_values = self.critic(ret_states).detach()
        ret_values_cpu = ret_values.cpu()

        summary = defaultdict(list)
        values_index_start = 0
        for game_id in game_index:
            episode_len = self.episode_len[game_id]

            actions = self.actions[game_id, :episode_len]
            log_probs = self.log_probs[game_id, :episode_len]
            rewards = self.rewards[game_id, :episode_len]
            next_values = self.next_values[game_id, episode_len-1]

            values = ret_values_cpu[values_index_start : values_index_start + episode_len]
            values_index_start += episode_len

            returns, gaes = calculate_rewards(values, rewards, next_values, self.max_episode_len, self.gamma, self.tau)

            returns = torch.tensor(returns).float()
            gaes = torch.tensor(gaes).float()

            ret_actions.append(actions)
            ret_log_probs.append(log_probs)
            ret_gaes.append(gaes)
            ret_returns.append(returns)

            exploration = self.explorations[game_id, :episode_len].float().sum() / float(episode_len) * 100

            summary['rewards'].append(rewards.sum())
            summary['exploration'].append(exploration)

        ret_actions = torch.cat(ret_actions, 0).to(self.device)
        ret_log_probs = torch.cat(ret_log_probs, 0).to(self.device)
        ret_gaes = torch.cat(ret_gaes, 0).to(self.device)
        ret_returns = torch.cat(ret_returns, 0).to(self.device)

        self.summary_writer.add_scalar(f'{self.summary_prefix}/episode_len', self.episode_len.float().mean(), self.global_step)
        self.summary_writer.add_scalar(f'{self.summary_prefix}/returns', torch.mean(ret_returns), self.global_step)
        self.summary_writer.add_scalar(f'{self.summary_prefix}/gaes', torch.mean(ret_gaes), self.global_step)
        self.summary_writer.add_scalar(f'{self.summary_prefix}/rewards', np.mean(summary['rewards']), self.global_step)
        self.summary_writer.add_scalar(f'{self.summary_prefix}/exploration', np.mean(summary['exploration']), self.global_step)
        self.summary_writer.add_histogram(f'{self.summary_prefix}/actions', ret_actions, self.global_step)
        self.summary_writer.add_histogram(f'{self.summary_prefix}/log_probs', ret_log_probs, self.global_step)

        return game_index, ret_states, ret_actions, ret_log_probs, ret_gaes, ret_values, ret_returns

    def step(self, player_id, game_index, actions):
        actions = actions.to(self.games.device)

        games = self.games[game_index]

        games, rewards, dones = step_games(games, player_id, actions, self.num_rows, self.num_columns, self.inarow)
        self.games[game_index] = games.detach().clone()
        #print(f'{self.games[0].detach().cpu().numpy().astype(int).reshape(self.num_rows, self.num_columns)}')

        states = self.make_states(player_id, games)
        return states, rewards, dones

    def close(self):
        pass
