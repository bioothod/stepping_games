import importlib
import os

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

default_config = {
    'checkpoint_path': 'submission.ckpt',
    'rows': 6,
    'columns': 7,
    'inarow': 4,
    'num_actions': 7,
    'batch_size': 128,
}

config_ppo6 = deepcopy(default_config)
config_ppo6.update({
    'num_features': 512,
    'hidden_dims': [128],
})
config_ppo8 = deepcopy(default_config)
config_ppo8.update({
    'num_features': 512,
    'hidden_dims': [128],
})
config_ppo9 = deepcopy(default_config)
config_ppo9.update({
    'num_features': 1024,
    'hidden_dims': [256],
})

def load_module_from_source(source_path):
    spec = importlib.util.spec_from_file_location('model', source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def create_actor(actor_source_path, feature_model_source_path, config, checkpoint_path):
    feature_module = load_module_from_source(feature_model_source_path)

    def feature_model_creation_func(config):
        return feature_module.Model(config)

    actor_module = load_module_from_source(actor_source_path)
    actor = actor_module.Actor(config, feature_model_creation_func)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.train(False)

    return actor

class CombinedModel(nn.Module):
    def __init__(self, is_local):
        self.num_actions = default_config['columns']

        kaggle_prefix = '/kaggle_simulations/agent/'
        model_paths = [
            ('rl_agents_ppo9.py', 'feature_model_ppo9.py', 'submission_9_ppo79.ckpt', config_ppo9),
        ]

        self.actors = []
        for rl_model_path, feature_model_path, checkpoint_path, config in model_paths:
            if not is_local:
                rl_model_path = os.path.join(kaggle_prefix, rl_model_path)
                feature_model_path = os.path.join(kaggle_prefix, feature_model_path)
                checkpoint_path = os.path.join(kaggle_prefix, checkpoint_path)

            actor = create_actor(rl_model_path, feature_model_path, config, checkpoint_path)
            self.actors.append(actor)

    def forward_from_observation(self, observation):
        all_probs = torch.ones((1, self.num_actions), dtype=torch.float32)
        for actor in self.actors:
            state = actor.create_state_from_observation(observation)
            states = state.unsqueeze(0)

            state_features = actor.state_features(states)
            logits = actor.features(state_features)
            probs = F.softmax(logits, 1)
            all_probs *= probs

        actions = torch.argmax(all_probs, 1)
        action = actions.squeeze(0).detach().cpu().numpy()
        return int(action)

want_test = os.environ.get('RUN_KAGGLE_TEST')

actor = CombinedModel(want_test)

if want_test:
    import kaggle_environments as kaggle
    env = kaggle.make('connectx')
    player_id = 1
    game = env.train([None, 'random'])

    from time import perf_counter

    start_time = perf_counter()
    steps = 0

    for game_idx in range(100):
        observation = game.reset()

        for i in range(100):
            action = actor.forward_from_observation(observation)
            observation, reward, done, info = game.step(action)
            steps += 1
            if done:
                break

    game_time = perf_counter() - start_time
    time_per_step_ms = game_time / steps * 1000

    print(f'time_per_step_ms: {time_per_step_ms:.1f}')

def my_agent(observation, config):
    action = actor.forward_from_observation(observation)
    return action
