import argparse
import importlib
import os

from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn

import evaluate
from logger import setup_logger

class EmptySummaryWriter:
    def __init__(self):
        pass

    def add_scalar(self, *args, **kwargs):
        pass
    def add_scalars(self, *args, **kwargs):
        pass
    def add_histogram(self, *args, **kwargs):
        pass
    def add_image(self, *args, **kwargs):
        pass
    def add_graph(self, *args, **kwargs):
        pass
    def add_audio(self, *args, **kwargs):
        pass

def create_actor(module_source_path, checkpoint_path):
    spec = importlib.util.spec_from_file_location('model', module_source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    default_config = module.default_config

    actor = module.Actor(default_config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.train(False)

    return actor

class AgentWrapper(nn.Module):
    def __init__(self, actor):
        super().__init__()

        self.actor = actor

    def set_training_mode(self, mode):
        self.actor.train(mode)

    def dist_actions(self, inputs):
        return self.actor.dist_actions(inputs)

    def forward(self, inputs):
        return self.actor.forward(inputs)

def create_agent(name, logger):
    split = name.split(':')
    if len(split) != 3:
        raise ValueError(f'invalid agent name: {name}, format: name:module_path:checkpoint_path')

    agent_name, module_path, checkpoint_path = split
    if len(module_path) == 0:
        actor = agent_name
        logger.info(f'using builtin agent \'{agent_name}\'')
        return actor

    actor = create_actor(module_path, checkpoint_path)
    agent = AgentWrapper(actor)
    return agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_agent', type=str, required=True, help='The training agent\'s name')
    parser.add_argument('--eval_agent', type=str, required=True, help='The evaluation agent\'s name')
    parser.add_argument('--evaluation_dir', type=str, required=True, help='Working directory')
    parser.add_argument('--num_evaluations', type=int, default=100, help='Number of evaluation runs')
    parser.add_argument('--log_to_stdout', action='store_true', help='Log evaluation data to stdout')
    parser.add_argument('--eval_seed', type=int, default=555, help='Random seed for generators')
    FLAGS = parser.parse_args()

    train_name = FLAGS.train_agent.split(':')[0]
    eval_name = FLAGS.eval_agent.split(':')[0]
    evaluation_dir = os.path.join(FLAGS.evaluation_dir, f'{train_name}_vs_{eval_name}')

    os.makedirs(evaluation_dir, exist_ok=True)
    logfile = os.path.join(evaluation_dir, 'evalution.log')
    logger = setup_logger('e', logfile=logfile, log_to_stdout=FLAGS.log_to_stdout)

    config = edict({
        'device': 'cpu',
        'rows': 6,
        'columns': 7,
        'inarow': 4,
        'player_ids': [1, 2],
        'eval_seed': FLAGS.eval_seed,
        'logfile': logfile,
        'log_to_stdout': FLAGS.log_to_stdout,
        'train_player_id': 1,

        'gamma': 0.99,
        'tau': 0.97,
        'batch_size': 128,
    })

    config.actions = config.columns
    config.max_episode_len = config.rows * config.columns

    summary_writer = EmptySummaryWriter()
    eval_global_step = torch.zeros(1).long()

    try:
        train_agent = create_agent(FLAGS.train_agent, logger)
        eval_agent = create_agent(FLAGS.eval_agent, logger)
    except Exception as e:
        logger.critical(f'could not create an agent: {e}')
        exit(-1)

    evaluation = evaluate.Evaluate(config, logger, FLAGS.num_evaluations, eval_agent, summary_writer, eval_global_step)
    wins, evaluation_rewards = evaluation.evaluate(train_agent)

    mean_evaluation_reward = np.mean(evaluation_rewards)

    logger.info(f'{FLAGS.train_agent} vs {FLAGS.eval_agent}: wins: {wins}, mean_evaluation_reward: {mean_evaluation_reward:.4f}')

    evaluation.close()

if __name__ == '__main__':
    main()
