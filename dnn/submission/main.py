import os

import torch

from model import Actor, default_config

default_config['checkpoint_path'] = '/kaggle_simulations/agent/submission.ckpt'

actor = Actor(default_config)
checkpoint = torch.load(default_config['checkpoint_path'], map_location='cpu')
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.train(False)

want_test = os.environ.get('RUN_KAGGLE_TEST')
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
