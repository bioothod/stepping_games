import argparse
import os

from easydict import EasyDict as edict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

import evaluate
from logger import setup_logger
from mcts_value import Runner
import submission.utils as sub_utils

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

class AgentWrapper(nn.Module):
    def __init__(self, config, logger, actor, critic, mcts_steps):
        super().__init__()

        self.logger = logger

        self.real_actor = actor
        self.actor = actor
        self.critic = critic

        if mcts_steps > 0:
            self.actor = Runner(config, actor, critic, logger)

    def set_training_mode(self, mode):
        self.real_actor.train(mode)

    def create_state(self, player_id, game_state):
        return self.real_actor.create_state(player_id, game_state)

    def dist_actions(self, player_id, game_states):
        return self.actor.dist_actions(player_id, game_states)

def create_agent(global_config, name, logger, mcts_steps):
    split = name.split(':')
    if len(split) != 4:
        raise ValueError(f'invalid agent name: {name}, format: name:feature_model_path:rl_model_path:checkpoint_path')

    create_critic = False
    if mcts_steps > 0:
        create_critic = True

    agent_name, feature_model_path, rl_model_path, checkpoint_path = split
    if len(feature_model_path) == 0:
        actor = agent_name
        logger.info(f'using builtin agent \'{agent_name}\'')
        return actor

    config = deepcopy(global_config)
    local_config = sub_utils.select_config_from_feature_model(feature_model_path)
    config.update(local_config)

    actor, critic = sub_utils.create_actor_critic(feature_model_path, rl_model_path, config, checkpoint_path, create_critic)
    agent = AgentWrapper(config, logger, actor, critic, mcts_steps)
    return agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_agent', type=str, required=True, help='The training agent\'s name')
    parser.add_argument('--eval_agent', type=str, required=True, help='The evaluation agent\'s name')
    parser.add_argument('--evaluation_dir', type=str, required=True, help='Working directory')
    parser.add_argument('--num_evaluations', type=int, default=100, help='Number of evaluation runs')
    parser.add_argument('--log_to_stdout', action='store_true', help='Log evaluation data to stdout')
    parser.add_argument('--mcts_steps', type=int, default=0, help='Wrap training agent into mcts tree search with this many rollouts per step')
    parser.add_argument('--eval_seed', type=int, default=555, help='Random seed for generators')
    FLAGS = parser.parse_args()

    train_name = FLAGS.train_agent.split(':')[0]
    eval_name = FLAGS.eval_agent.split(':')[0]
    evaluation_dir = os.path.join(FLAGS.evaluation_dir, f'{train_name}_vs_{eval_name}')

    os.makedirs(evaluation_dir, exist_ok=True)
    logfile = os.path.join(evaluation_dir, 'evalution.log')
    logger = setup_logger('e', logfile=logfile, log_to_stdout=FLAGS.log_to_stdout)

    summary_writer = EmptySummaryWriter()
    eval_global_step = torch.zeros(1).long()

    config = edict({
        'device': 'cpu',
        'rows': 6,
        'columns': 7,
        'inarow': 4,
        'player_ids': [1, 2],
        'eval_seed': FLAGS.eval_seed,
        'logfile': logfile,
        'log_to_stdout': FLAGS.log_to_stdout,
        'num_training_games': FLAGS.num_evaluations,
        'default_reward': 0,

        'gamma': 0.99,
        'tau': 0.97,
        'batch_size': 128,

        'num_simulations': FLAGS.mcts_steps,
        'mcts_c1': 1.25,
        'mcts_c2': 19652,
        'mcts_discount': 0.99,
        'add_exploration_noise': False,
        'root_dirichlet_alpha': 0.3,
        'root_exploration_fraction': 0.25,

    })

    config.actions = config.columns
    config.max_episode_len = config.rows * config.columns

    try:
        train_agent = create_agent(config, FLAGS.train_agent, logger, FLAGS.mcts_steps)
        eval_agent = create_agent(config, FLAGS.eval_agent, logger, 0)
    except Exception as e:
        logger.critical(f'could not create an agent: {e}')
        raise

    for train_player_id in config.player_ids:
        config.train_player_id = train_player_id

        evaluation = evaluate.Evaluate(config, logger, FLAGS.num_evaluations, eval_agent, summary_writer, eval_global_step)
        wins, evaluation_rewards = evaluation.evaluate(train_agent)

        mean_evaluation_reward = np.mean(evaluation_rewards)

        logger.info(f'{FLAGS.train_agent} vs {FLAGS.eval_agent}: train_player_id: {train_player_id}, wins: {wins}, mean_evaluation_reward: {mean_evaluation_reward:.4f}')

        evaluation.close()

if __name__ == '__main__':
    main()
