from copy import deepcopy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from mcts_value import Runner

class CombinedModel(nn.Module):
    def __init__(self, is_local):
        super().__init__()

        self.num_actions = utils.default_config['columns']

        kaggle_prefix = '/kaggle_simulations/agent/'
        model_paths = [
            #('rl_agents_ppo9_multichannel.py', 'feature_model_ppo9_multichannel.py', 'submission_9_ppo86_multichannel_critic.ckpt', utils.config_ppo9_multichannel),
            ('rl_agents_ppo18.py', 'feature_model_ppo18.py', 'checkpoint_ppo_18_best_score_77.7.ckpt', utils.config_ppo18),
            ('rl_agents_ppo12.py', 'feature_model_ppo12.py', 'submission_12_ppo83_critic.ckpt', utils.config_ppo12),
        ]

        self.actors = []
        self.critic = None
        self.actor_for_state = None
        for rl_model_path, feature_model_path, checkpoint_path, config in model_paths:
            if not is_local:
                rl_model_path = os.path.join(kaggle_prefix, rl_model_path)
                feature_model_path = os.path.join(kaggle_prefix, feature_model_path)
                checkpoint_path = os.path.join(kaggle_prefix, checkpoint_path)

            create_critic = False
            if rl_model_path.endswith('ppo12.py'):
                create_critic = True

            actor, critic = utils.create_actor_critic(feature_model_path, rl_model_path, config, checkpoint_path, create_critic)
            self.actors.append(actor)

            if create_critic:
                self.actor_for_state = actor
                self.critic = critic

    def create_game_from_state(self, player_id, state):
        raise NotImplementedError(f'combined_mode::create_game_from_state: player_id: {player_id}, state: {state}')

    def create_state(self, player_id, game_state):
        return self.actor_for_state.create_state(player_id, game_state)

    def forward(self, player_id, game_state):
        all_probs = torch.ones((1, self.num_actions), dtype=torch.float32)
        for actor in self.actors:
            state = actor.create_state(player_id, game_state)

            state_features = actor.state_features(state)
            logits = actor.features(state_features)
            probs = F.softmax(logits, 1)
            all_probs *= probs

        return torch.log(all_probs)

want_test = os.environ.get('RUN_KAGGLE_TEST')

actor = CombinedModel(want_test)
class CriticWrapper(nn.Module):
    def __init__(self, combined_model):
        super().__init__()

        self.combined_model = combined_model

    def forward(self, state):
        return self.combined_model.critic(state)

mcts_config = deepcopy(utils.default_config)
mcts_config.update({
    'num_simulations': 5,
    'num_training_games': 1,
    'mcts_c1': 1.25,
    'mcts_c2': 19652,
    'mcts_discount': 0.99,
    'add_exploration_noise': False,
    'root_dirichlet_alpha': 0.3,
    'root_exploration_fraction': 0.25,
    'device': 'cpu',
    'default_reward': 0,
})
critic = CriticWrapper(actor)


def create_game_from_observation(obs, config):
    orig_state = np.asarray(obs['board'], dtype=np.float32).reshape(1, config['rows'], config['columns'])

    state = torch.from_numpy(orig_state)
    player_id = obs['mark']

    return player_id, state

from logger import setup_logger
logger = setup_logger('test', None, True)

mcts_actors = {player_id:Runner(mcts_config, actor, critic, logger=logger) for player_id in [1, 2]}

import connectx_impl
game_state = torch.zeros((1, 1, mcts_config['rows'], mcts_config['columns'])).float()
actions = torch.tensor([1]).long()
player_id = int(1)
num_rows = int(mcts_config['rows'])
num_columns = int(mcts_config['columns'])
num_inarow = int(mcts_config['inarow'])
default_reward = 0.0
new_game_state, rewards, dones = connectx_impl.step_games(game_state, player_id, actions, num_rows, num_columns, num_inarow, default_reward)


def my_agent(observation, config):
    player_id, game_state = create_game_from_observation(observation, config)
    actions = mcts_actors[player_id].forward(player_id, game_state)
    return actions[0]

if want_test:
    import kaggle_environments as kaggle
    env = kaggle.make('connectx')
    player_id = 1
    game = env.train([None, 'random'])

    from time import perf_counter

    start_time = perf_counter()
    steps = 0

    logger = None

    for game_idx in range(100):
        observation = game.reset()

        for i in range(100):
            action = my_agent(observation, mcts_config)
            observation, reward, done, info = game.step(action)
            steps += 1
            if done:
                break

    game_time = perf_counter() - start_time
    time_per_step_ms = game_time / steps * 1000

    print(f'time_per_step_ms: {time_per_step_ms:.1f}')
