import itertools
import os
import random

from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn

from .. import logger
from ..agents import base_agent

import action_strategies
import ddqn
import gym_env
import models
import replay_buffer

class FullModel(nn.Module):
    def __init__(self, config, feature_model_creation_func, action_model_creation_func):
        self.feature_model = feature_model_creation_func(config)
        self.action_model = action_model_creation_func(config)

    def forward(self, inputs):
        features = self.feature_model(inputs)
        actions = self.action_model(features)
        return actions

class ModelWrapper(base_agent.Agent):
    def __init__(self, name, config, feature_model_creation_func, action_model_creation_func):
        super().__init__(name, config)

        self.gamma = config.gamma
        self.tau = config.tau
        
        self.model = FullModel(config, feature_model_creation_func, action_model_creation_func).to(config.device)
        self.target_model = FullModel(config, feature_model_creation_func, action_model_creation_func).to(config.device)

        self.update_network(tau=1.0)

        self.max_gradient_norm = 1
        self.value_opt = torch.optim.Adam(self.model.parameters(), lr=config.init_lr)

    def train(self, batch):
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
            target.data.copy(mixed_weights)

    def action(self, state):
        action = self.model(state)
        return action
    
class Trainer:
    def __init__(self):
        self.seed = 444
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.checkpoints_dir = 'checkpoints'
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.checkpoint_path = os.path.join(self.checkpoints_dir, 'latest.ckpt')

        logfile = os.path.join(self.checkpoints_dir, 'ddqn.log')
        self.logger = logger.setup_logger('ddqn', logfile, log_to_stdout=True)

        self.max_eval_reward = 0.0
        
        device = 'cuda:0'
        #device = 'cpu'
        self.device = torch.device(device)

        self.config = edict({
            'rows': 6,
            'columns': 7,
            'inarow': 4,
            'init_lr': 1e-3,
            'min_lr': 1e-5,
            'device': self.device,

            'gamma': 0.99,
            'tau': 0.1

            'num_features': 64,
        })

        self.num_warmup_batches = 10
        self.batch_size = 128

        def feature_model_creation_func(config):
            model = models.simple_model.Model(config)
            return model

        def action_model_creation_func(config):
            model = ddqn.DDQN(config)
            return model
        
        self.agent1 = ModelWrapper('ddqn', self.config, feature_model_creation_func, action_model_creation_func)
        self.agent2 = 'negamax'

        self.env = gym_env.ConnectXGym(self.config, self.agent1, self.agent2)

        self.replay_buffer = replay_buffer.ReplayBuffer(obs_shape=self.env.observation_space,
                                                        obs_dtype=self.env.observation_dtype,
                                                        action_shape=self.env.action_shape,
                                                        capacity=50000,
                                                        device=device)

        self.action_strategy = action_strategies.EGreedyExpStrategy()

    def evaluate(self, num_episodes=1):
        rewards = []
        for _ in range(num_episodes):
            rewards.append(0)

            state = self.env.reset()
            done = False

            while not done:
                action1 = self.agent1(state)
                new_state, reward, done, info = self.env.step(action1)
                rewards[-1] += reward

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        return mean_reward, std_reward

    def run_epoch(self):
        episode_reward = []

        state = self.env.reset()
        done = False
        while not done:
            action1 = self.action_strategy.select_action(self.agent1, state)
            new_state, reward, done, info = self.env.step(action1)
            is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
            is_failure = done and not is_truncated

            self.replay_buffer.add(state, action1, new_state, reward, float(is_failure))
            episode_reward.append(reward)

            if len(self.replay_buffer) > self.batch_size * self.num_warmup_batches:
                experiences = self.replay_buffer.sample(self.batch_size)
                self.agent1.train(experiences)

                self.agent1.update_network()

        mean_eval_reward, std_eval_reward = self.evaluate()
        self.logger.info(f'mean_eval_reward: {mean_eval_reward:.2f}, max_reward: {self.max_eval_reward:.2f}, episode_length: {len(episode_reward)}')

        if mean_eval_reward >= self.max_eval_reward:
            self.max_eval_reward = mean_eval_reward

            torch.save({
                'model_state_dict': self.agent1.state_dict(),
                'mean_eval_reward': mean_eval_reward,
                }, self.checkpoint_path)



def main():
    trainer = Trainer()
    for epoch in itertools.count():
        trainer.run_epoch()

        
if __name__ == '__main__':
    main()
