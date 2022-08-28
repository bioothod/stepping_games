import itertools
import os
import random

from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn

import gym

import action_strategies
import ddqn
import gym_env
from multiprocess_env import MultiprocessEnv
import networks
import logger
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
            'checkpoints_dir': 'checkpoints_cnn',
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

        self.max_eval_metric = 0.0

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

        self.agent1 = ModelWrapper('ddqn', self.config, feature_model_creation_func, action_model_creation_func, self.logger)
        self.agent2 = 'negamax'

        self.try_load('agent1', self.agent1)

        make_args_fn = lambda: {}
        def make_env_fn(seed=None):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            return gym_env.ConnectXGym(self.config, self.agent2)
        
        train_num_workers = 32
        eval_num_workers = 8
        self.train_env = MultiprocessEnv('train', make_env_fn, make_args_fn, self.config, self.train_seed, train_num_workers)
        self.eval_env = MultiprocessEnv('eval', make_env_fn, make_args_fn, self.config, self.eval_seed, eval_num_workers)
        #self.env = gym_env.ConnectXGym(self.config, self.agent2)
        #self.env = gym.make('CartPole-v1')

        state = self.train_env.reset(None)[0, :, :, :]
        self.replay_buffer = replay_buffer.ReplayBuffer(obs_shape=state.shape,
                                                        obs_dtype=state.dtype,
                                                        action_shape=(self.config.num_actions, ),
                                                        capacity=100000,
                                                        device=device)

        self.episode_reward = []
        self.episode_timestep = []
        self.evaluation_scores = []
        self.episode_exploration = []

    def evaluate(self, num_episodes=1):
        self.agent1.set_training_mode(False)
        evaluation_rewards = []
        for _ in range(num_episodes):
            states = self.eval_env.reset()
            worker_ids = self.eval_env.worker_ids

            batch_size = len(states)
            evaluation_rewards += [0.0] * batch_size

            while True:
                actions1 = self.eval_action_strategy.select_action(self.agent1, states)
                new_states, rewards, dones, infos = self.eval_env.step(worker_ids, actions1)
                
                #self.logger.info(f'eval: worker_ids: {worker_ids}, actions: {actions1}, rewards: {rewards}, dones: {dones}')

                ready_states = []
                ret_worker_ids = []
                for enum_idx, (worker_id, new_state, reward, done) in enumerate(zip(worker_ids, new_states, rewards, dones)):
                    if not done:
                        ready_states.append(new_state)
                        ret_worker_ids.append(worker_id)
                        
                    evaluation_rewards[-1 - worker_id] += reward

                if len(ready_states) == 0:
                    break

                states = np.array(ready_states, dtype=np.float32)
                worker_ids = ret_worker_ids

        return evaluation_rewards

    def try_train(self):
        if len(self.replay_buffer) < self.config.batch_size * self.config.num_warmup_batches:
            return False

        new_batch_size = self.config.batch_size + 1024
        if len(self.replay_buffer) >= new_batch_size and new_batch_size < 1024*10:
            self.logger.info(f'train: batch_size update: {self.config.batch_size} -> {new_batch_size}')
            self.config.batch_size = new_batch_size
            
        self.agent1.set_training_mode(True)
        experiences = self.replay_buffer.sample(self.config.batch_size)
        self.agent1.train(experiences)
        self.agent1.update_network()

        return True

    def add_flipped_state(self, state, action, reward, new_state, done):
        state_flipped = np.flip(state, 2)
        action_flipped = self.config.num_actions - action - 1
        new_state_flipped = np.flip(new_state, 2)
        self.replay_buffer.add(state_flipped, action_flipped, reward, new_state_flipped, done)

    def add_opposite_state(self, state, action, reward, new_state, done):
        def make_opposite(state):
            state_opposite = state.copy()
            state_opposite[state == 1] = 2
            state_opposite[state == 2] = 1
            return state_opposite

        state_opposite = make_opposite(state)
        new_state_opposite = make_opposite(new_state)
        if reward == -1:
            reward = 1
        elif reward == 1:
            reward = -1
        self.replay_buffer.add(state_opposite, action_opposite, reward, new_state_opposite, done)

    def run_epoch(self, epoch):
        training_started = False
        
        states = self.train_env.reset()
        worker_ids = self.train_env.worker_ids

        batch_size = len(states)
        
        self.episode_reward += [0.0] * batch_size
        self.episode_timestep += [0] * batch_size
        self.episode_exploration += [0.0] * batch_size

        while True:
            self.agent1.set_training_mode(False)

            actions1 = self.train_action_strategy.select_action(self.agent1, states)
            new_states, rewards, dones, infos = self.train_env.step(worker_ids, actions1)

            ret_states = []
            ret_worker_ids = []
            for enum_idx, (worker_id, state, action1, reward, new_state, done, info) in enumerate(zip(worker_ids, states, actions1, rewards, new_states, dones, infos)):
                is_truncated = info.get('TimeLimit.truncated', False)
                is_failure = done and not is_truncated
                is_failure = float(is_failure)

                self.replay_buffer.add(state, action1, reward, new_state, is_failure)

                if not done:
                    ret_states.append(new_state)
                    ret_worker_ids.append(worker_id)
                    
                idx = -1 - worker_id
                self.episode_reward[idx] += reward
                self.episode_timestep[idx] += 1
                self.episode_exploration[idx] += int(self.train_action_strategy.exploratory_action_taken[enum_idx])

                self.add_flipped_state(state, action1, reward, new_state, is_failure)

            training_started = self.try_train()
            
            states = np.array(ret_states)
            worker_ids = ret_worker_ids
            
            if len(ret_states) == 0:
                break

        eval_rewards = self.evaluate()
        self.evaluation_scores += eval_rewards

        total_step = int(np.sum(self.episode_timestep))
        mean_10_reward = np.mean(self.episode_reward[-10:])
        std_10_reward = np.std(self.episode_reward[-10:])
        mean_100_reward = np.mean(self.episode_reward[-100:])
        std_100_reward = np.std(self.episode_reward[-100:])
        
        mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
        std_100_eval_score = np.std(self.evaluation_scores[-100:])
        mean_10_eval_score = np.mean(self.evaluation_scores[-10:])
        std_10_eval_score = np.std(self.evaluation_scores[-10:])

        lst_10_exp_rat = np.array(self.episode_exploration[-10:])/np.array(self.episode_timestep[-10:])
        mean_10_exp_rat = np.mean(lst_10_exp_rat)
        std_10_exp_rat = np.std(lst_10_exp_rat)

        wins_100 = int(np.count_nonzero(np.array(self.evaluation_scores[-100:]) >= 1) / len(self.evaluation_scores[-100:]) * 100)
        eval_metric = wins_100

        if training_started:
            self.logger.info(f'{total_step:6d}: epoch: {epoch:4d}, episode: len: {self.episode_timestep[-1]:2d}, '
                             f'reward: {self.episode_reward[-1]:6.3f}, '
                             f'e10: {mean_10_reward:6.3f}\u00B1{std_10_reward:5.3f}, '
                             f'e100: {mean_100_reward:6.3f}\u00B1{std_100_reward:5.3f}, '
                             f'eval_score: '
                             f'e100: {mean_100_eval_score:6.3f}\u00B1{std_100_eval_score:5.3f}, '
                             f'e10: {mean_10_eval_score:6.3f}\u00B1{std_10_eval_score:5.3f}, '
                             f'wins100: {wins_100:2d}%, '
                             f'max_eval_metric: {self.max_eval_metric:2d}, '
                             f'last10_exploration: {mean_10_exp_rat:.3f}\u00B1{std_10_exp_rat:.3f}'
                             )

        if eval_metric >= self.max_eval_metric:
            self.max_eval_metric = eval_metric

            checkpoint_path = os.path.join(self.config.checkpoints_dir, f'agent1_{eval_metric}.ckpt')
            self.agent1.save(checkpoint_path)
            self.logger.info(f'eval_metric: {eval_metric:2d}, saved agent1 -> {checkpoint_path}')

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
            if len(metric) != 2:
                continue

            try:
                metric = float(metric[1])
            except:
                continue

            if max_metric is None or metric > max_metric:
                checkpoint_path = os.path.join(self.config.checkpoints_dir, checkpoint_fn)
                max_metric = metric

        if checkpoint_path is not None:
            self.logger.info(f'{name}: loading checkpoint {checkpoint_path}, metric: {max_metric}')
            model.load(checkpoint_path)
            self.max_eval_metric = max_metric
        
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
