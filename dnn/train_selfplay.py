import itertools
import os
import random

from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True

import action_strategies
import connectx_impl
import gym_env
import logger
from multiprocess_env import MultiprocessEnv
import networks

class BaseTrainer:
    def __init__(self):
        self.train_seed = 444
        self.eval_seed = 555
        
        torch.manual_seed(self.train_seed)
        np.random.seed(self.train_seed)
        random.seed(self.train_seed)

        device = 'cuda:0'
        #device = 'cpu'

        self.config = edict({
            'checkpoints_dir': 'checkpoints_simple3_selfplay_3',
            'device': torch.device(device),
            
            'rows': 6,
            'columns': 7,
            'inarow': 4,

            'init_lr': 1e-4,
            'min_lr': 1e-5,

            'gamma': 0.99,
            'tau': 0.1,

            'num_features': 512,
            'num_actions': 7,

            'num_warmup_batches': 1,
            'batch_size': 1024,
            'max_batch_size': 1024*32,

            'train_num_games': 1024,

            'hidden_dims': [128],
        })
        self.config.observation_shape = [1, self.config.rows, self.config.columns]

        os.makedirs(self.config.checkpoints_dir, exist_ok=True)

        self.config.logfile = os.path.join(self.config.checkpoints_dir, 'ddqn.log')
        self.config.log_to_stdout = True
        self.logger = logger.setup_logger('ddqn', self.config.logfile, log_to_stdout=self.config.log_to_stdout)

        config_message = []
        for k, v in self.config.items():
            config_message.append(f'{k:>32s}:{str(v):>16s}')
        config_message = '\n'.join(config_message)

        self.logger.info(f'config:\n{config_message}')

        self.eval_action_strategy = action_strategies.GreedyStrategy()
        self.eval_agent = 'negamax'

        self.train_env = connectx_impl.ConnectX(self.config, self.config.train_num_games, replay_buffer=None)

        make_args_fn = lambda: {}
        def make_env_fn(seed=None):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            pair = [None, self.eval_agent]
            return gym_env.ConnectXGym(self.config, pair)

        eval_num_workers = 50
        self.eval_env = MultiprocessEnv('eval', make_env_fn, make_args_fn, self.config, self.eval_seed, eval_num_workers)

        self.num_evaluations_per_epoch = 100
        self.evaluation_scores = []

        self.max_eval_metric = 0.0

    def evaluate(self, train_agent):
        train_agent.set_training_mode(False)
        evaluation_rewards = []
        
        while len(evaluation_rewards) < self.num_evaluations_per_epoch:
            states = self.eval_env.reset()
            worker_ids = self.eval_env.worker_ids

            batch_size = len(states)
            episode_rewards = [0.0] * batch_size

            while True:
                actions, _ = self.eval_action_strategy.select_action(train_agent, states)
                new_states, rewards, dones, infos = self.eval_env.step(worker_ids, actions)

                ready_states = []
                ret_worker_ids = []
                for worker_id, new_state, reward, done in zip(worker_ids, new_states, rewards, dones):
                    if not done:
                        ready_states.append(new_state)
                        ret_worker_ids.append(worker_id)
                        
                    episode_rewards[worker_id] += reward

                if len(ready_states) == 0:
                    break

                states = np.array(ready_states, dtype=np.float32)
                worker_ids = ret_worker_ids

            evaluation_rewards += episode_rewards

        last = evaluation_rewards[-self.num_evaluations_per_epoch:]
        wins = np.count_nonzero(np.array(last) >= 1) / len(last) * 100
        
        return wins, evaluation_rewards

    def try_train(self):
        raise NotImplementedError('method @try_train() needs to be implemented')

    def run_epoch(self, train_agent):
        training_started = False

        for _ in range(100):
            train_agent.set_training_mode(False)

            training_started = self.try_train()

        if training_started:
            mean_rewards, std_rewards, mean_expl, std_expl, mean_timesteps, std_timesteps = self.train_env.completed_games_stats(100)
            if len(mean_rewards) == 0:
                return

            last_game_stat = self.train_env.last_game_stats()

            eval_metric, eval_rewards = self.evaluate(train_agent)
            self.evaluation_scores += eval_rewards

            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])

            wins100 = int(np.count_nonzero(np.array(self.evaluation_scores[-100:]) >= 1) / len(self.evaluation_scores[-100:]) * 100)

            self.logger.info(f'{self.train_env.total_steps:6d}: '
                             f'games: {len(self.train_env.completed_games):5d}, '
                             f'last: '
                             f'ts: {last_game_stat.player_stats[1].timesteps:2d}, '
                             f'r: {last_game_stat.player_stats[1].reward:5.2f} / {last_game_stat.player_stats[2].reward:5.2f}, '
                             f'last100: '
                             f'ts: {mean_timesteps[1]:4.1f}\u00B1{std_timesteps[1]:3.1f} / {mean_timesteps[2]:4.1f}\u00B1{std_timesteps[2]:3.1f}, '
                             f'r: {mean_rewards[1]:5.2f}\u00B1{std_rewards[1]:4.2f} / {mean_rewards[2]:5.2f}\u00B1{std_rewards[2]:4.2f}, '
                             f'expl: {mean_expl[1]:.2f}\u00B1{std_expl[1]:.1f} / {mean_expl[2]:.2f}\u00B1{std_expl[2]:.1f}, '
                             f'eval: '
                             f'r100: {mean_100_eval_score:5.2f}\u00B1{std_100_eval_score:4.2f}, '
                             f'metric: {eval_metric:2.0f} / {self.max_eval_metric:2.0f}'
                             )

            if eval_metric > self.max_eval_metric:
                self.max_eval_metric = eval_metric

                checkpoint_path = os.path.join(self.config.checkpoints_dir, f'{train_agent.name}_{eval_metric:.0f}.ckpt')
                train_agent.save(checkpoint_path)
                self.logger.info(f'eval_metric: {eval_metric:.0f}, saved {train_agent.name} -> {checkpoint_path}')

    def try_load(self, name, model):
        max_metric = None
        checkpoint_path = None
        
        for checkpoint_fn in os.listdir(self.config.checkpoints_dir):
            if checkpoint_fn == f'{name}.ckpt':
                checkpoint_path = os.path.join(self.config.checkpoints_dir, checkpoint_fn)
                max_metric = 0.0
                continue

            metric, ext = os.path.splitext(checkpoint_fn)
            if ext != '.ckpt':
                continue

            if not metric.startswith(name):
                continue

            metric = metric.split('_')
            try:
                metric = float(metric[-1])
            except:
                continue

            if max_metric is None or metric > max_metric:
                checkpoint_path = os.path.join(self.config.checkpoints_dir, checkpoint_fn)
                max_metric = metric

        if checkpoint_path is not None and max_metric is not None:
            self.logger.info(f'{name}: loading checkpoint {checkpoint_path}, metric: {max_metric}')
            model.load(checkpoint_path)
            self.max_eval_metric = max_metric
            return True

        return False
        
    def stop(self):
        self.eval_env.close()
