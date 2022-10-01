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

    def actions(self, states):
        rollouts_states = []
        for state in states:
            rollouts_state = state.unsqueeze(0)
            rollouts_state = torch.tile(rollouts_state, [self.mcts.num_rollouts_per_game, 1, 1, 1])
            rollouts_states.append(rollouts_state)

        rollouts_states = torch.cat(rollouts_states)
        mcts_actions = self.mcts.run(self.player_id, self.agent, rollouts_states)
        mcts_actions = mcts_actions.to(states.device)

        return mcts_actions

    def greedy_actions(self, states):
        return self.actions(states)

    def dist_actions(self, states):
        actions, log_probs, explorations = self.agent.dist_actions(states)
        mcts_actions = self.actions(states)

        changed = torch.count_nonzero(actions != mcts_actions) / len(actions) * 100
        self.logger.info(f'mcts: actions_changed: {changed:.1f}%')

        return mcts_actions, log_probs, explorations

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

    def run(self, train_player_id, train_agent, states):
        self.env.reset()

        index = torch.arange(len(states))
        self.env.set_index_state(index, states)

        while True:
            for player_id in self.config.player_ids:
                game_index, states = self.env.current_states(player_id)
                if len(game_index) == 0:
                    break

                states = states.to(self.config.device)
                actions, log_probs, explorations = train_agent.dist_actions(states)

                states, rewards, dones = self.env.step(player_id, game_index, actions)

                self.env.update_game_rewards(player_id, game_index, states, actions, log_probs, rewards, dones, torch.zeros_like(rewards), explorations)

            running_index = self.env.running_index()
            if len(running_index) == 0:
                break


        max_actions = []

        rollout = defaultdict(float)
        items_in_rollout = 0
        for game_id in index:
            player_stats = self.env.player_stats[train_player_id]
            episode_len = player_stats.episode_len[game_id]

            reward = player_stats.rewards[game_id, :episode_len].sum()
            action = player_stats.actions[game_id][0]

            rollout[action] += reward
            items_in_rollout += 1
            if items_in_rollout == self.num_rollouts_per_game:
                max_action = max(rollout, key=rollout.get)
                max_actions.append(max_action)
                rollout.clear()
                items_in_rollout = 0

        max_actions = torch.tensor(max_actions).long()
        return max_actions
