import importlib

from copy import deepcopy

import torch

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

def select_config_from_feature_model(feature_model_path):
    if feature_model_path.endswith('ppo6.py'):
        config = config_ppo6
    elif feature_model_path.endswith('ppo8.py'):
        config = config_ppo8
    elif feature_model_path.endswith('ppo9.py'):
        config = config_ppo9
    else:
        raise ValueError(f'there is no matching config for the feature model path {feature_model_path}, please check how the name ends, it should end with ppo6.py or something similar')

    return config

def load_module_from_source(source_path):
    spec = importlib.util.spec_from_file_location('model', source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def create_actor(feature_model_path, rl_model_path, config, checkpoint_path):
    feature_module = load_module_from_source(feature_model_path)

    def feature_model_creation_func(config):
        return feature_module.Model(config)

    actor_module = load_module_from_source(rl_model_path)
    actor = actor_module.Actor(config, feature_model_creation_func)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.train(False)

    return actor
