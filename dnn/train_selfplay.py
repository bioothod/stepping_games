import itertools
import os
import random

from collections import defaultdict
from easydict import EasyDict as edict
from kaggle_environments import perf_counter

import numpy as np
import torch
import torch.nn as nn

import action_strategies
import connectx_impl
import gym_env
import ddqn
import logger
from multiprocess_env import MultiprocessEnv
import networks
import replay_buffer

class FullModel(nn.Module):
    def __init__(self, config, feature_model_creation_func, action_model_creation_func):
        super().__init__()

        self.feature_model = feature_model_creation_func(config)
        self.action_model = action_model_creation_func(config)

    def forward(self, inputs):
        features = self.feature_model(inputs)
        actions = self.action_model(features)
        return actions

class ModelWrapper:
    def __init__(self, name, config, feature_model_creation_func, action_model_creation_func, logger):
        #super().__init__(name, config)
        self.logger = logger
        self.name = name

        self.device = config.device
        self.gamma = config.gamma
        self.tau = config.tau
        
        self.model = FullModel(config, feature_model_creation_func, action_model_creation_func).to(config.device)
        self.target_model = FullModel(config, feature_model_creation_func, action_model_creation_func).to(config.device)
        self.target_model.train(False)

        self.update_network(tau=1.0)

        self.max_gradient_norm = 1
        self.value_opt = torch.optim.Adam(self.model.parameters(), lr=config.init_lr)

    def set_training_mode(self, training):
        self.model.train(training)
        
    def train(self, batch):
        self.model.zero_grad()
        self.model.train()

        states, actions, rewards, next_states, is_terminals = batch
        batch_size = len(is_terminals)

        argmax_a_q_sp = self.model(next_states).max(1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)

        q_sa = self.model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()

        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
        self.value_opt.step()

    def update_network(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_model.parameters(), self.model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data

            mixed_weights = target_ratio + online_ratio
            target.data = mixed_weights.detach().clone()

    def save(self, checkpoint_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.value_opt.state_dict(),
            }, checkpoint_path)

        return checkpoint_path

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.value_opt.load_state_dict(checkpoint['optimizer_state_dict'])

        self.logger.info(f'{self.name}: loaded checkpoint {checkpoint_path}')

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return x

    def __call__(self, state):
        state = self._format(state)

        action = self.model(state)
        return action
    
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
            'checkpoints_dir': 'checkpoints_cnn_selfplay',
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
        })
        
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)

        self.config.logfile = os.path.join(self.config.checkpoints_dir, 'ddqn.log')
        self.config.log_to_stdout = True
        self.logger = logger.setup_logger('ddqn', self.config.logfile, log_to_stdout=self.config.log_to_stdout)

        def feature_model_creation_func(config):
            #model = networks.simple_model.Model(config)
            #model = networks.simple2_model.Model(config)
            model = networks.conv_model.Model(config)
            #model = networks.empty_model.Model(config)
            return model

        def action_model_creation_func(config):
            model = ddqn.DDQN(config)
            return model

        self.train_action_strategy = action_strategies.EGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.01, decay_steps=30000)
        self.eval_action_strategy = action_strategies.GreedyStrategy()

        self.train_agent = ModelWrapper('ddqn', self.config, feature_model_creation_func, action_model_creation_func, self.logger)
        self.eval_agent = 'negamax'

        self.train_agent_name = 'selfplay_agent'
        model_loaded = self.try_load(self.train_agent_name, self.train_agent)

        train_num_games = 1024*2
        self.train_env = connectx_impl.ConnectX(self.config, train_num_games)

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

        self.replay_buffer = replay_buffer.ReplayBuffer(obs_shape=self.train_env.observation_shape,
                                                        obs_dtype=np.float32,
                                                        action_shape=(self.config.num_actions, ),
                                                        capacity=100000,
                                                        device=device)

        self.episode_reward = defaultdict(list)
        self.episode_timestep = defaultdict(list)
        self.episode_exploration = defaultdict(list)
        self.evaluation_scores = []

        self.max_eval_metric = 0.0
        if model_loaded:
            self.max_eval_metric, _ = self.evaluate()
            self.logger.info(f'initial evaluation metric: {self.max_eval_metric:.2f}')

    def evaluate(self):
        self.train_agent.set_training_mode(False)
        evaluation_rewards = []
        
        while len(evaluation_rewards) < self.num_evaluations_per_epoch:
            states = self.eval_env.reset()
            worker_ids = self.eval_env.worker_ids

            batch_size = len(states)
            episode_rewards = [0.0] * batch_size

            while True:
                actions = self.eval_action_strategy.select_action(self.train_agent, states)
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
        if len(self.replay_buffer) < self.config.batch_size * self.config.num_warmup_batches:
            return False

        new_batch_size = self.config.batch_size + 1024
        if len(self.replay_buffer) >= new_batch_size and new_batch_size <= 1024*32:
            self.logger.info(f'train: batch_size update: {self.config.batch_size} -> {new_batch_size}')
            self.config.batch_size = new_batch_size
            
        self.train_agent.set_training_mode(True)
        experiences = self.replay_buffer.sample(self.config.batch_size)
        self.train_agent.train(experiences)
        self.train_agent.update_network()

        return True

    def add_flipped_state(self, state, action, reward, new_state, done):
        state_flipped = np.flip(state, 2)
        action_flipped = self.config.num_actions - action - 1
        new_state_flipped = np.flip(new_state, 2)
        self.replay_buffer.add(state_flipped, action_flipped, reward, new_state_flipped, done)

    def make_opposite(self, state):
        state_opposite = state.detach().clone()
        state_opposite[state == 1] = 2
        state_opposite[state == 2] = 1
        return state_opposite

    def add_experiences(self, player_id, states, actions, rewards, new_states, dones):
        if player_id == 2:
            states = self.make_opposite(states)
            new_states = self.make_opposite(new_states)
        
        states = states.detach().cpu().numpy()
        new_states = new_states.detach().cpu().numpy()
        
        for state, action, reward, new_state, done in zip(states, actions, rewards, new_states, dones):
            if player_id == 2:
                if reward == -1:
                    reward = 1
                elif reward == 1:
                    reward = -1

            self.replay_buffer.add(state, action, reward, new_state, float(done))
            self.add_flipped_state(state, action, reward, new_state, done)

    def select_games(self, player_id, rewards, dones):
        reset_ids = []
        for game_id, (reward, done) in enumerate(zip(rewards, dones)):
            if done:
                reset_ids.append(game_id)

            idx = -1 - game_id
            self.episode_reward[player_id][idx] += reward
            self.episode_timestep[player_id][idx] += 1
            self.episode_exploration[player_id][idx] += int(self.train_action_strategy.exploratory_action_taken[game_id])

        return reset_ids
    
    def make_step(self):
        states = self.train_env.current_games()
        states = states.to(self.config.device)
        actions0 = self.train_action_strategy.select_action(self.train_agent, states)
        actions0 = torch.from_numpy(actions0)
        new_states, rewards, dones = self.train_env.step(1, actions0)
        self.add_experiences(1, states, actions0, rewards, new_states, dones)

        reset_ids = self.select_games(0, rewards, dones)
        self.train_env.reset_games(reset_ids)
        
        states1 = self.train_env.current_games()
        opposite_states1 = self.make_opposite(states1)
        opposite_states1 = opposite_states1.to(self.config.device)
        # agent's network assumes inputs are always related to player0
        actions1 = self.train_action_strategy.select_action(self.train_agent, opposite_states1)

        actions1 = torch.from_numpy(actions1)
        new_states1, rewards1, dones1 = self.train_env.step(2, actions1)
        self.add_experiences(2, states1, actions1, rewards1, new_states1, dones1)
        
        reset_ids = self.select_games(1, rewards1, dones1)
        self.train_env.reset_games(reset_ids)
        
    def run_epoch(self, epoch):
        training_started = False
        
        states = self.train_env.reset()
        batch_size = len(states)

        for player_id in range(2):
            self.episode_reward[player_id] += [0.0] * batch_size
            self.episode_timestep[player_id] += [0] * batch_size
            self.episode_exploration[player_id] += [0.0] * batch_size

        for _ in range(100):
            self.train_agent.set_training_mode(False)

            self.make_step()
            training_started = self.try_train()

        eval_start_time = perf_counter()
        eval_metric, eval_rewards = self.evaluate()
        eval_time = perf_counter() - eval_start_time
        
        self.evaluation_scores += eval_rewards

        total_step = int(np.sum(self.episode_timestep[0]))
        mean_100_reward0 = np.mean(self.episode_reward[0][-100:])
        std_100_reward0 = np.std(self.episode_reward[0][-100:])
        mean_100_reward1 = np.mean(self.episode_reward[1][-100:])
        std_100_reward1 = np.std(self.episode_reward[1][-100:])
        
        mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
        std_100_eval_score = np.std(self.evaluation_scores[-100:])

        lst_10_exp_rat0 = np.array(self.episode_exploration[0][-10:])/np.array(self.episode_timestep[0][-10:])
        mean_10_exp_rat0 = np.mean(lst_10_exp_rat0)
        std_10_exp_rat0 = np.std(lst_10_exp_rat0)
        lst_10_exp_rat1 = np.array(self.episode_exploration[1][-10:])/np.array(self.episode_timestep[1][-10:])
        mean_10_exp_rat1 = np.mean(lst_10_exp_rat1)
        std_10_exp_rat1 = np.std(lst_10_exp_rat1)

        wins100 = int(np.count_nonzero(np.array(self.evaluation_scores[-100:]) >= 1) / len(self.evaluation_scores[-100:]) * 100)

        if training_started:
            self.logger.info(f'{total_step:6d}: '
                             f'epoch: {epoch:4d}, '
                             f'episode: len: {self.episode_timestep[0][-1]:2d} / {self.episode_timestep[1][-1]:2d}, '
                             f'reward: {self.episode_reward[0][-1]:5.2f} / {self.episode_reward[1][-1]:5.2f}, '
                             f'e100: {mean_100_reward0:5.2f}\u00B1{std_100_reward0:4.2f} / {mean_100_reward1:4.2f}\u00B1{std_100_reward1:4.2f}, '
                             f'eval_score: '
                             f'e100: {mean_100_eval_score:5.2f}\u00B1{std_100_eval_score:4.2f}, '
                             f'wins100: {wins100:2d}, '
                             f'eval_metric: {eval_metric:2.0f} / {self.max_eval_metric:2.0f}, '
                             f'eval_time: total: {eval_time:.1f}, '
                             f'expl10: {mean_10_exp_rat0:.3f}\u00B1{std_10_exp_rat0:.3f} / {mean_10_exp_rat1:.3f}\u00B1{std_10_exp_rat1:.3f}'
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
