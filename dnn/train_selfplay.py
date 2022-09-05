import itertools
import os
import random

from collections import defaultdict
from copy import deepcopy
from easydict import EasyDict as edict
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True

import action_strategies
import connectx_impl
import gym_env
import ddqn
import logger
from multiprocess_env import MultiprocessEnv
import networks
import replay_buffer

class Trainer:
    def __init__(self):
        self.train_seed = 444
        self.eval_seed = 555
        
        torch.manual_seed(self.train_seed)
        np.random.seed(self.train_seed)
        random.seed(self.train_seed)

        device = 'cuda:0'
        #device = 'cpu'

        self.config = edict({
            'checkpoints_dir': 'checkpoints_simple3_selfplay_4',
            'device': torch.device(device),
            
            'rows': 6,
            'columns': 7,
            'inarow': 4,

            'init_lr': 1e-4,
            'min_lr': 1e-5,

            'gamma': 0.99,
            'tau': 0.1,

            'player_ids': [1, 2],
            'replay_buffer_size': 100_000_000,

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

        def feature_model_creation_func(config):
            #model = networks.simple_model.Model(config)
            #model = networks.simple2_model.Model(config)
            model = networks.simple3_model.Model(config)
            #model = networks.conv_model.Model(config)
            #model = networks.empty_model.Model(config)
            return model

        self.train_action_strategy = action_strategies.EGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.1, decay_steps=300_000_000)
        self.eval_action_strategy = action_strategies.GreedyStrategy()

        self.train_agent = ddqn.DDQN('ddqn', self.config, feature_model_creation_func, self.logger)
        self.eval_agent = 'negamax'

        self.train_agent_name = 'selfplay_agent'
        model_loaded = self.try_load(self.train_agent_name, self.train_agent)

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
        if model_loaded:
            eval_time_start = perf_counter()
            self.max_eval_metric, _ = self.evaluate()
            eval_time = perf_counter() - eval_time_start

            self.logger.info(f'initial evaluation metric: {self.max_eval_metric:.2f}, evaluation time: {eval_time:.1f} sec')

    def evaluate(self):
        self.train_agent.set_training_mode(False)
        evaluation_rewards = []
        
        while len(evaluation_rewards) < self.num_evaluations_per_epoch:
            states = self.eval_env.reset()
            worker_ids = self.eval_env.worker_ids

            batch_size = len(states)
            episode_rewards = [0.0] * batch_size

            while True:
                actions, _ = self.eval_action_strategy.select_action(self.train_agent, states)
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

    def make_step(self):
        for player_id in self.config.player_ids:
            rewards, dones, explorations = self.model.make_single_step_and_save(player_id)
            self.train_env.update_game_rewards(player_id, rewards, dones, explorations)

        training_started = self.model.try_train()
        return training_started

    def run_epoch(self, epoch):
        training_started = False

        for _ in range(100):
            self.train_agent.set_training_mode(False)
            training_started = self.make_step()

        if training_started:
            mean_rewards, std_rewards, mean_expl, std_expl, mean_timesteps, std_timesteps = self.train_env.completed_games_stats(100)
            if len(mean_rewards) == 0:
                return

            last_game_stat = self.train_env.last_game_stats()

            eval_metric, eval_rewards = self.evaluate()
            self.evaluation_scores += eval_rewards

            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])

            wins100 = int(np.count_nonzero(np.array(self.evaluation_scores[-100:]) >= 1) / len(self.evaluation_scores[-100:]) * 100)

            self.logger.info(f'{self.train_env.total_steps:6d}: '
                             f'games: {len(self.train_env.completed_games):5d}, '
                             f'buffer%: {self.replay_buffer.filled()*100:2.0f}, '
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

                checkpoint_path = os.path.join(self.config.checkpoints_dir, f'{self.train_agent_name}_{eval_metric:.0f}.ckpt')
                self.train_agent.save(checkpoint_path)
                self.logger.info(f'eval_metric: {eval_metric:.0f}, saved {self.train_agent_name} -> {checkpoint_path}')

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
        self.train_env.close()
        self.eval_env.close()

def main():
    trainer = Trainer()
    for epoch in itertools.count():
        try:
            trainer.run_epoch(epoch)
        except Exception as e:
            print(f'type: {type(e)}, exception: {e}')
            trainer.stop()

            if type(e) != KeyboardInterrupt:
                raise

        
if __name__ == '__main__':
    main()
