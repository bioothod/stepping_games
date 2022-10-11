from typing import *

import joblib
import math

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

import connectx_impl

class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Node:
    def __init__(self, player_id, prior, log_prior):
        self.player_id = player_id
        self.prior = prior
        self.log_prior = log_prior
        self.game_state: torch.Tensor = None

        self.done = False
        self.value_sum: float = 0
        self.reward = 0
        self.visit_count = 0
        self.children = {}

    def __str__(self):
        return (
            f'player_id: {self.player_id}, '
            f'prior: {self.prior:.3f}, '
            f'visit_count: {self.visit_count}, '
            f'expanded: {self.expanded()}, '
            f'reward: {self.reward:.3f}, '
            f'value_sum: {self.value_sum:.3f}, '
            f'value: {self.value():.3f}'
        )

    def expanded(self):
        return len(self.children) > 0

    def is_done(self):
        return self.done

    def set_done(self):
        self.done = True

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, other_player_id, policy_logits, reward):
        self.reward = reward
        probs = torch.softmax(policy_logits, 0)
        self.probs = probs

        for action, (prob, log_prob) in enumerate(zip(probs, policy_logits)):
            self.children[action] = Node(other_player_id, prob, log_prob)

    def set_state(self, game_state):
        self.game_state = game_state

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MCTSValue:
    def __init__(self, player_id, config, root, game_state, actor, critic, logger):
        self.num_simulations = config['num_simulations']
        self.mcts_c1 = config['mcts_c1']
        self.mcts_c2 = config['mcts_c2']
        self.mcts_discount = config['mcts_discount']
        self.root_dirichlet_alpha = config['root_dirichlet_alpha']
        self.root_exploration_fraction = config['root_exploration_fraction']
        self.rows = config['rows']
        self.columns = config['columns']
        self.inarow = config['inarow']
        self.device = config['device']
        self.default_reward = config['default_reward']
        self.player_ids = config['player_ids']

        self.logger = logger

        self.actor = actor
        self.critic = critic

        self.min_max_stats = MinMaxStats()
        self.max_tree_depth = 0

        if root is None:
            root = Node(player_id, 0, 0)

        self.root : Node = root
        self.root.set_state(game_state)

    def select_child(self, node: Node):
        """
        Select the child with the highest UCB score.
        """

        max_ucb = -float('inf')
        max_children = []
        for action, child in node.children.items():
            child_score = self.ucb_score(node, child)
            if child_score > max_ucb:
                max_ucb = child_score
                max_children = [(action, child)]
            elif child_score == max_ucb:
                max_children.append((action, child))

        num_max_children = len(max_children)
        if num_max_children == 1:
            max_child_index = 0
        else:
            max_child_index = np.random.choice(num_max_children)

        action, child = max_children[max_child_index]
        return action, child

    def ucb_score(self, parent: Node, child: Node):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        score = math.log((parent.visit_count + self.mcts_c2 + 1) / self.mcts_c2) + self.mcts_c1
        score *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = score * child.prior

        if child.visit_count > 0:
            child_value = child.value()
            if parent.player_id != child.player_id:
                child_value = -child_value

            value_score = self.min_max_stats.normalize(child.reward + self.mcts_discount * child_value)
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, player_id, search_path: List[Node], value: float):
        for node in reversed(search_path):
            node_mult = 1
            if player_id != node.player_id:
               node_mult = -1

            node.value_sum += value * node_mult
            node.visit_count += 1
            self.min_max_stats.update(node.reward + self.mcts_discount * -node.value())

            value = node_mult * node.reward + self.mcts_discount * value

    def node_expand(self, parent: Node, node: Node, action: int):
        with torch.no_grad():
            game_state = parent.game_state.detach().clone()
            game_state = game_state.to(self.device)
            actions = torch.Tensor([action]).long()

            #self.logger.info(f'node_expand: parent: {parent.player_id}, node: {node.player_id}, actions: {actions}, game_state: {game_state.shape}')
            new_game_state, rewards, dones = connectx_impl.step_games(game_state, node.player_id, actions, self.rows, self.columns, self.inarow, self.default_reward)

            new_state = self.actor.create_state(node.player_id, new_game_state)
            new_state = new_state.to(self.device)
            new_value = self.critic(new_state)
            new_value = new_value.detach().cpu()

            logits = self.actor.forward(node.player_id, new_game_state)

            node_player_index = self.player_ids.index(node.player_id)
            node_player_index += 1
            node_player_index %= len(self.player_ids)

            other_player_id = self.player_ids[node_player_index]
            node.expand(other_player_id, logits[0], rewards[0])

            if dones[0]:
                node.set_done()

            #self.logger.info(f' parent: {parent}, node: {node}, new_value: {new_value}, logits: {logits}, actions: {actions}, game_state:\n{new_game_state}')
            node.set_state(new_game_state)

        return node, new_value[0]

    def expand_root(self, node):
        with torch.no_grad():
            logits = self.actor.forward(node.player_id, node.game_state)
            node.expand(node.player_id, logits[0], 0)

    def run(self, add_exploration_noise: bool):
        if not self.root.expanded():
            self.expand_root(self.root)

        if add_exploration_noise:
            self.root.add_exploration_noise(dirichlet_alpha=self.root_dirichlet_alpha, exploration_fraction=self.root_exploration_fraction)

        max_tree_depth = 0
        for simulation_idx in range(self.num_simulations):
            tree_depth = 0
            action = 0
            node = self.root
            search_path = [node]

            #self.logger.info(f'{simulation_idx}/{self.num_simulations}: start')

            while node.expanded():
                tree_depth += 1

                action, node = self.select_child(node)
                #self.logger.info(f'  tree_depth: {tree_depth}, action: {action}, child: {node}, child_expanded: {node.expanded()}')
                search_path.append(node)

            if node.is_done():
                continue

            player_id = node.player_id
            parent = search_path[-2]
            node, value = self.node_expand(parent, node, action)

            self.backpropagate(player_id, search_path, value)

            max_tree_depth = max(tree_depth, max_tree_depth)

        action, node = self.select_child(self.root)
        return action, node.log_prior

        max_visit_count = 0
        max_action = -1
        max_node: Node = None
        for action, node in self.root.children.items():
            if node.visit_count > max_visit_count:
                max_visit_count = node.visit_count
                max_action = action
                max_node = node


        return max_action, max_node.log_prior

class Game:
    def __init__(self, game_id, config, start_player_id, actor, critic, logger):
        self.game_id = game_id
        self.logger = logger

        self.actor = actor
        self.critic = critic

        self.config = deepcopy(config)
        self.config['num_training_games'] = 1

        self.start_player_id = start_player_id
        self.mcts : MCTSValue = None

    def reset(self, player_id, root, game_state):
        self.mcts = MCTSValue(player_id, self.config, root, game_state, self.actor, self.critic, self.logger)

    def step(self, player_id, game_state, add_exploration_noise):
        #self.logger.info(f'mcts::game::step: game_id: {self.game_id}, start_player_id: {self.start_player_id}, game_state: {game_state.shape}')

        self.reset(player_id, None, game_state)

        #self.logger.info(f'game_id: {self.game_id}, start_player_id: {self.start_player_id}, game_state:\n{game_state}')
        action, log_prob = self.mcts.run(add_exploration_noise=add_exploration_noise)

        self.mcts.root = self.mcts.root.children[action]

        return action, log_prob

class Runner(nn.Module):
    def __init__(self, config, player_id, actor, critic, logger):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.config = config
        self.logger = logger
        self.player_id = player_id
        self.games = [Game(game_id, config, player_id, actor, critic, logger) for game_id in range(config['num_training_games'])]

    def step(self, player_id, game_states):
        if True:
            jobs = []
            for game_id, (game, game_state) in enumerate(zip(self.games, game_states)):
                game_state = game_state.unsqueeze(0)
                job = joblib.delayed(game.step)(player_id, game_state, add_exploration_noise=self.config.add_exploration_noise)
                jobs.append(job)

            with joblib.parallel_backend('threading', n_jobs=16):
                results = joblib.Parallel(require='sharedmem')(jobs)
        else:
            #self.logger.info(f'runner::step: player_id: {player_id}, game_states: {game_states.shape}, games: {len(self.games)}')
            results = self.games[0].step(player_id, game_states, add_exploration_noise=False)
            action, log_prob = results
            self.logger.info(f'results: {results}')
            return [action], [log_prob]

        all_actions = []
        all_probs = []
        for action, prob in results:
            all_actions.append(action)
            all_probs.append(prob)

        return all_actions, all_probs

    def create_state(self, player_id, game_states):
        return self.actor.create_state(player_id, game_states)

    def greedy_actions(self, player_id, game_states):
        mcts_actions, mcts_log_probs = self.step(player_id, game_states)
        return mcts_actions

    def forward(self, player_id, game_state):
        game_state = game_state.unsqueeze(0)
        return self.games[0].step(player_id, game_state, add_exploration_noise=False)

    def dist_actions(self, player_id, game_states):
        actions, log_probs, explorations = self.actor.dist_actions(player_id, game_states)

        mcts_actions, mcts_log_probs = self.step(player_id, game_states)
        mcts_actions = torch.tensor(mcts_actions).long().to(actions.device)
        mcts_log_probs = torch.tensor(mcts_log_probs).float().to(log_probs.device)

        changed_mcts = torch.count_nonzero(actions != mcts_actions) / len(mcts_actions) * 100

        self.logger.info(f'mcts: player_id: {player_id}, game_states: {game_states.shape}, changed_mcts: {changed_mcts:.1f}%')

        return mcts_actions, mcts_log_probs, explorations
