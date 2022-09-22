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
import logger
import networks

class BaseTrainer:
    def __init__(self, config):
        config.train_seed = 444
        config.eval_seed = 555

        torch.manual_seed(config.train_seed)
        np.random.seed(config.train_seed)
        random.seed(config.train_seed)

        config.device = torch.device('cuda:0')
        config.rows = 6
        config.columns = 7
        config.inarow = 4
        config.player_ids = [1, 2]
        config.num_actions = config.columns
        config.observation_shape = [1, self.config.rows, self.config.columns]

        self.config = config

        os.makedirs(self.config.checkpoints_dir, exist_ok=True)

        self.config.logfile = os.path.join(self.config.checkpoints_dir, 'train.log')
        self.config.log_to_stdout = True
        self.logger = logger.setup_logger('t', self.config.logfile, log_to_stdout=self.config.log_to_stdout)

        config_message = []
        for k, v in self.config.items():
            config_message.append(f'{k:>32s}:{str(v):>16s}')
        config_message = '\n'.join(config_message)

        self.logger.info(f'config:\n{config_message}')

        self.num_evaluations_per_epoch = 100
        self.evaluation_scores = []
        self.max_eval_metric = 0.0

        if not config.get('eval_checkpoint_path'):
            import gym_env
            from multiprocess_env import MultiprocessEnv

            self.eval_agent_name = 'negamax'

            make_args_fn = lambda: {}
            def make_env_fn(seed=None):
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                pair = [None, self.eval_agent_name]

                return gym_env.ConnectXGym(config, pair)

            eval_num_workers = 50
            self.eval_env = MultiprocessEnv('eval', make_env_fn, make_args_fn, self.config, self.config.eval_seed, eval_num_workers)
            self.evaluate = lambda agent: self.evaluate_multiprocess(agent)

        else:
            from submission.model import Actor as eval_agent_model
            import connectx_impl

            self.eval_agent = eval_agent_model(config)
            checkpoint = torch.load(config.eval_checkpoint_path, map_location='cpu')
            self.eval_agent.load_state_dict(checkpoint['actor_state_dict'])
            self.eval_agent.train(False)

            self.eval_agent_name = config.eval_checkpoint_path

            self.eval_env = connectx_impl.ConnectX(config, self.num_evaluations_per_epoch)
            self.evaluate = lambda agent: self.evaluate_torch(agent)


    def evaluate_multiprocess(self, train_agent):
        train_agent.set_training_mode(False)
        evaluation_rewards = []
        
        while len(evaluation_rewards) < self.num_evaluations_per_epoch:
            states = self.eval_env.reset()
            worker_ids = self.eval_env.worker_ids

            batch_size = len(states)
            episode_rewards = [0.0] * batch_size

            while True:
                states = torch.from_numpy(states).to(self.config.device)

                actions = train_agent.actor.greedy_actions(states)
                actions = actions.detach().cpu().numpy()
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
        wins = int(np.count_nonzero(np.array(last) >= 1) / len(last) * 100)
        
        return wins, evaluation_rewards

    def evaluate_torch(self, train_agent):
        train_agent.set_training_mode(False)

        train_player_id = 1
        eval_player_id = 2

        self.eval_env.reset()

        while True:
            for player_id in [train_player_id, eval_player_id]:
                game_index, states = self.eval_env.current_states()

                if player_id == 2:
                    states = train_agent.make_opposite(states)

                if player_id == train_player_id:
                    states = states.to(self.config.device)
                    actions, log_probs, explorations = train_agent.actor.dist_actions(states)
                else:
                    actions, log_probs, explorations = self.eval_agent.dist_actions(states)

                states, rewards, dones = self.eval_env.step(player_id, game_index, actions)

                self.eval_env.update_game_rewards(player_id, game_index, states, actions, log_probs, rewards, dones, explorations, torch.zeros_like(rewards))

            completed_index = self.eval_env.completed_index()
            if len(completed_index) >= self.num_evaluations_per_epoch:
                break

        completed_index = self.eval_env.completed_index()

        player_idx = self.config.player_ids.index(train_player_id)

        evaluation_rewards = []
        for game_id in completed_index:
            episode_len = self.eval_env.episode_len[game_id]
            reward = self.eval_env.rewards[game_id, :episode_len, player_idx].sum()
            evaluation_rewards.append(reward)

        wins = int(np.count_nonzero(np.array(evaluation_rewards) >= 1) / len(evaluation_rewards) * 100)

        return wins, evaluation_rewards

    def try_train(self):
        raise NotImplementedError('method @try_train() needs to be implemented')

    def run_epoch(self, train_env, train_agent):
        training_started = False

        for _ in range(self.config.eval_after_train_steps):
            train_agent.set_training_mode(False)

            training_started = self.try_train()

        if training_started:
            mean_rewards, std_rewards, mean_explorations, std_explorations, mean_timesteps, std_timesteps = train_env.completed_games_stats()
            if len(mean_rewards) == 0:
                return

            last_timesteps, last_rewards = train_env.last_game_stats()

            eval_metric, eval_rewards = self.evaluate(train_agent)
            self.evaluation_scores += eval_rewards

            mean_eval_score = np.mean(self.evaluation_scores)
            std_eval_score = np.std(self.evaluation_scores)

            mean_eval_score = np.mean(self.evaluation_scores)

            wins100 = int(np.count_nonzero(np.array(self.evaluation_scores[-100:]) >= 1) / len(self.evaluation_scores[-100:]) * 100)

            self.logger.info(f'games: {train_env.total_games_completed:6d}: '
                             f'last: '
                             f'ts: {last_timesteps:2d}, '
                             f'r: {last_rewards[0]:5.2f} / {last_rewards[1]:5.2f}, '
                             f'last: '
                             f'ts: {mean_timesteps:4.1f}\u00B1{std_timesteps:3.1f}, '
                             f'r: {mean_rewards[0]:5.2f}\u00B1{std_rewards[0]:4.2f} / {mean_rewards[1]:5.2f}\u00B1{std_rewards[1]:4.2f}, '
                             f'expl: {mean_explorations[0]:.2f}\u00B1{std_explorations[0]:.1f} / {mean_explorations[1]:.2f}\u00B1{std_explorations[1]:.1f}, '
                             f'eval: '
                             f'r: {mean_eval_score:5.2f}\u00B1{std_eval_score:4.2f}, '
                             f'mean: {mean_eval_score:6.3f}, '
                             f'metric: {eval_metric:2.0f} / {self.max_eval_metric:2.0f}'
                             )

            if eval_metric > 0 and eval_metric >= self.max_eval_metric:
                if eval_metric < 100:
                    self.max_eval_metric = eval_metric

                checkpoint_path = os.path.join(self.config.checkpoints_dir, f'{train_agent.name}_{eval_metric:.0f}.ckpt')
                train_agent.save(checkpoint_path)
                self.logger.info(f'eval_metric: {eval_metric:.0f}, saved {train_agent.name} -> {checkpoint_path}')

    def try_load(self, name, model):
        max_metric = None
        checkpoint_path = None

        checkpoints_dir = self.config.get('load_checkpoints_dir', self.config['checkpoints_dir'])
        for checkpoint_fn in os.listdir(checkpoints_dir):
            if checkpoint_fn == f'{name}.ckpt':
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_fn)
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
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_fn)
                max_metric = metric

        if checkpoint_path is not None and max_metric is not None:
            self.logger.info(f'{name}: loading checkpoint {checkpoint_path}, metric: {max_metric}')
            model.load(checkpoint_path)
            self.max_eval_metric = max_metric
            return True

        return False
        
    def stop(self):
        self.eval_env.close()
