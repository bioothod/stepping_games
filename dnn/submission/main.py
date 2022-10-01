import importlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import default_config

def create_actor(module_source_path, config, checkpoint_path):
    spec = importlib.util.spec_from_file_location('model', module_source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    actor = module.Actor(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.train(False)

    return actor

class CombinedModel(nn.Module):
    def __init__(self, config, is_local):
        self.num_actions = config['columns']

        kaggle_prefix = '/kaggle_simulations/agent/'
        model_paths = [
            ('model.py', 'submission.ckpt'),
            ('model.py', 'submission_8_ppo56.ckpt'),
        ]

        self.actors = []
        for model_path, checkpoint_path in model_paths:
            if not is_local:
                model_path = os.path.join(kaggle_prefix, model_path)
                checkpoint_path = os.path.join(kaggle_prefix, checkpoint_path)

            actor = create_actor(model_path, config, checkpoint_path)
            self.actors.append(actor)

    def forward(self, observation):
        all_probs = torch.ones((1, self.num_actions), dtype=torch.float32)
        for actor in self.actors:
            state = actor.create_state(observation)
            state = torch.from_numpy(state)
            states = state.unsqueeze(0)

            state_features = actor.state_features(states)
            logits = actor.features(state_features)
            probs = F.softmax(logits, 1)
            all_probs *= probs

        actions = torch.argmax(all_probs, 1)
        action = actions.squeeze(0).detach().cpu().numpy()
        return int(action)

want_test = os.environ.get('RUN_KAGGLE_TEST')

actor = CombinedModel(default_config, want_test)

if want_test:
    import kaggle_environments as kaggle
    env = kaggle.make('connectx')
    game = env.train([None, 'random'])

    from time import perf_counter

    start_time = perf_counter()
    steps = 0

    for game_idx in range(100):
        state = game.reset()

        for i in range(100):
            action = actor.forward(state)
            state, reward, done, info = game.step(action)
            steps += 1
            if done:
                break

    game_time = perf_counter() - start_time
    rps = game_time / steps
    print(rps)

def my_agent(observation, config):
    action = actor.forward(observation)
    return action
