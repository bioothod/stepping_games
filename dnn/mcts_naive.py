import math

from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

import connectx_impl

class MCTSWrapper(nn.Module):
    def __init__(self, player_id, agent, mcts_impl):
        super().__init__()

        self.player_id = player_id
        self.agent = agent
        self.mcts = mcts_impl
        self.logger = mcts_impl.logger

    def actions(self, player_states):
        game_states = self.agent.create_game_from_state(self.player_id, player_states)
        rollouts_states = []
        for state in game_states:
            rollouts_state = state.unsqueeze(0)
            rollouts_state = torch.tile(rollouts_state, [self.mcts.num_rollouts_per_game, 1, 1, 1])
            rollouts_states.append(rollouts_state)

        rollouts_states = torch.cat(rollouts_states)
        mcts_actions_probs = self.mcts.run(self.player_id, self.agent, rollouts_states)

        return mcts_actions_probs

    def create_state(self, player_id, game_states):
        return self.agent.create_state(player_id, game_states)

    def greedy_actions(self, states):
        return self.actions(states)

    def dist_actions(self, states):
        actions, log_probs, explorations = self.agent.dist_actions(states)

        if states.shape[1] == 1:
            used_space = torch.count_nonzero(states, (1, 2, 3))
        else:
            used_space = torch.count_nonzero(states[:, 1:, :, :], (1, 2, 3))

        used_index = used_space >= 0
        if used_index.sum() == 0:
            return actions, log_probs, explorations

        mcts_states = states[used_index]
        mcts_actions_probs = self.actions(mcts_states)
        max_actions = []
        for rollout, rollout_probs in mcts_actions_probs:
            max_action = max(rollout, key=rollout.get)
            max_actions.append(max_action)
        max_actions = torch.tensor(max_actions).long().to(actions.device)

        updated_actions = torch.where(used_index, max_actions, actions)

        changed_mcts = torch.count_nonzero(actions[used_index] != max_actions) / len(max_actions) * 100
        changed_final = torch.count_nonzero(actions != updated_actions) / len(actions) * 100
        self.logger.info(f'mcts: average episode_len: {float(torch.mean(used_space.float())):.1f}, '
                         f'using_mcts_for: {used_index.sum()}/{(len(states))} {used_index.sum()/len(states)*100:.1f}%, '
                         f'changed_mcts: {changed_mcts:.1f}%, '
                         f'changed_final: {changed_final:.1f}%')

        return updated_actions, log_probs, explorations

class MCTSNaive:
    def __init__(self, config, logger, num_rollouts_per_game):
        self.config = config
        self.logger = logger

        self.num_rollouts_per_game = num_rollouts_per_game

        self.num_players = len(self.config.player_ids)
        self.num_games = self.config.num_training_games * self.num_rollouts_per_game

        self.observation_shape = [1, self.config.rows, self.config.columns]
        self.observation_dtype = torch.float32

        local_config = deepcopy(config)
        local_config.num_training_games = self.num_games
        local_config.device = 'cpu'

        self.env = connectx_impl.ConnectX(local_config, critic=None, summary_writer=None, summary_prefix='', global_step=None)

    def run(self, train_player_id, train_agent, initial_game_states):
        self.env.reset()

        index = torch.arange(len(initial_game_states))
        self.env.set_index_state(index, initial_game_states)

        player_ids = self.config.player_ids * len(self.config.player_ids)
        player_index = self.config.player_ids.index(train_player_id)
        player_ids = player_ids[player_index : player_index + len(self.config.player_ids)]

        while True:
            for player_id in player_ids:
                game_index, game_states = self.env.current_states()
                if len(game_index) == 0:
                    break

                states = train_agent.create_state(player_id, game_states)
                states = states.to(self.config.device)
                actions, log_probs, explorations = train_agent.dist_actions(states)

                new_states, rewards, dones = self.env.step(player_id, game_index, actions)
                # in the line above 'new_states' becomes game state, we should save previous state here, but since it will not be used, we can save a new state instead
                states = new_states
                self.env.update_game_rewards(player_id, game_index, states, actions, log_probs, rewards, dones, torch.zeros_like(rewards), explorations)

            running_index = self.env.running_index()
            if len(running_index) == 0:
                break

        player_stats = self.env.player_stats[train_player_id]

        actions_probs = []
        rollout = defaultdict(float)
        rollout_probs = defaultdict(float)
        items_in_rollout = 0
        for game_id in index:
            episode_len = player_stats.episode_len[game_id]
            rewards = player_stats.rewards[game_id, :episode_len]

            if episode_len == 0:
                self.logger.info(f'player_id: {train_player_id}, game_id: {game_id}, episode_len: {episode_len}, rewards: {rewards}')
            reward = rewards[-1]
            action = player_stats.actions[game_id, 0]
            log_prob = player_stats.log_probs[game_id, 0]

            rollout[action] += reward
            rollout_probs[action] += log_prob

            items_in_rollout += 1
            if items_in_rollout == self.num_rollouts_per_game:
                actions_probs.append((deepcopy(rollout), deepcopy(rollout_probs)))

                rollout.clear()
                rollout_probs.clear()
                items_in_rollout = 0

        return actions_probs
