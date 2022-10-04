import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class CombinedModel(nn.Module):
    def __init__(self, is_local):
        self.num_actions = utils.default_config['columns']

        kaggle_prefix = '/kaggle_simulations/agent/'
        model_paths = [
            ('rl_agents_ppo6.py', 'feature_model_ppo6.py', 'submission_6_ppo100.ckpt', utils.config_ppo6),
            ('rl_agents_ppo9_multichannel.py', 'feature_model_ppo9_multichannel.py', 'submission_9_ppo72_multichannel.ckpt', utils.config_ppo9_multichannel),
        ]

        self.actors = []
        for rl_model_path, feature_model_path, checkpoint_path, config in model_paths:
            if not is_local:
                rl_model_path = os.path.join(kaggle_prefix, rl_model_path)
                feature_model_path = os.path.join(kaggle_prefix, feature_model_path)
                checkpoint_path = os.path.join(kaggle_prefix, checkpoint_path)

            actor = utils.create_actor(feature_model_path, rl_model_path, config, checkpoint_path)
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
