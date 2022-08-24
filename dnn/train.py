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
    def __init__(self, name, config, feature_model_creation_func, action_model_creation_func):
        #super().__init__(name, config)

        self.device = config.device
        self.gamma = config.gamma
        self.tau = config.tau
        
        self.model = FullModel(config, feature_model_creation_func, action_model_creation_func).to(config.device)
        self.target_model = FullModel(config, feature_model_creation_func, action_model_creation_func).to(config.device)

        self.update_network(tau=1.0)

        self.max_gradient_norm = 1
        self.value_opt = torch.optim.Adam(self.model.parameters(), lr=config.init_lr)

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

    def save(self, checkpoint_dir, name):
        checkpoint_path = os.path.join(checkpoint_dir, f'{name}.ckpt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            }, checkpoint_path)


    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def __call__(self, state):
        state = self._format(state)

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

            'init_lr': 1e-4,
            'min_lr': 1e-5,
            'device': self.device,

            'gamma': 0.999,
            'tau': 0.1,

            'num_features': 4,
            'num_actions': 2,
        })

        self.num_warmup_batches = 10
        self.batch_size = 128

        def feature_model_creation_func(config):
            #model = networks.simple_model.Model(config)
            return networks.empty_model.Model(config)

        def action_model_creation_func(config):
            model = ddqn.DDQN(config)
            return model
        
        self.agent1 = ModelWrapper('ddqn', self.config, feature_model_creation_func, action_model_creation_func)
        self.agent2 = 'negamax'

        #self.env = gym_env.ConnectXGym(self.config, self.agent1, self.agent2)
        self.env = gym.make('CartPole-v1')

        state = self.env.reset()
        self.replay_buffer = replay_buffer.ReplayBuffer(obs_shape=state.shape,
                                                        obs_dtype=state.dtype,
                                                        action_shape=(self.config.num_actions, ),
                                                        capacity=50000,
                                                        device=device)

        self.train_action_strategy = action_strategies.EGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.3, decay_steps=20000)
        self.eval_action_strategy = action_strategies.GreedyStrategy()

        self.episode_reward = []
        self.episode_timestep = []
        self.evaluation_scores = []
        self.episode_exploration = []

    def evaluate(self, num_episodes=1):
        rewards = []
        for _ in range(num_episodes):
            rewards.append(0)

            state = self.env.reset()
            done = False

            while not done:
                action1 = self.eval_action_strategy.select_action(self.agent1, state)
                new_state, reward, done, info = self.env.step(action1)
                state = new_state
                rewards[-1] += reward

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        return mean_reward, std_reward

    def run_epoch(self, epoch):
        self.episode_reward.append(0.0)
        self.episode_timestep.append(0.0)
        self.episode_exploration.append(0.0)

        state = self.env.reset()
        done = False
        while not done:
            action1 = self.train_action_strategy.select_action(self.agent1, state)
            new_state, reward, done, info = self.env.step(action1)
            is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
            is_failure = done and not is_truncated

            self.replay_buffer.add(state, action1, reward, new_state, float(is_failure))
            self.episode_reward[-1] += reward
            self.episode_timestep[-1] += 1
            self.episode_exploration[-1] += int(self.train_action_strategy.exploratory_action_taken)

            state = new_state

            if len(self.replay_buffer) > self.batch_size * self.num_warmup_batches:
                experiences = self.replay_buffer.sample(self.batch_size)
                self.agent1.train(experiences)
                self.agent1.update_network()

        mean_eval_reward, std_eval_reward = self.evaluate()
        self.evaluation_scores.append(mean_eval_reward)

        total_step = int(np.sum(self.episode_timestep))
        mean_10_reward = np.mean(self.episode_reward[-10:])
        std_10_reward = np.std(self.episode_reward[-10:])
        mean_100_reward = np.mean(self.episode_reward[-100:])
        std_100_reward = np.std(self.episode_reward[-100:])
        mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
        std_100_eval_score = np.std(self.evaluation_scores[-100:])

        lst_100_exp_rat = np.array(self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
        mean_100_exp_rat = np.mean(lst_100_exp_rat)
        std_100_exp_rat = np.std(lst_100_exp_rat)

        if total_step > self.num_warmup_batches * self.batch_size:
            self.logger.info(f'{total_step:6d}: epoch: {epoch:4d}, '
                             f'last10_reward: {mean_10_reward:5.1f}\u00B1{std_10_reward:5.1f}, '
                             f'last100_reward: {mean_100_reward:5.1f}\u00B1{std_100_reward:5.1f}, '
                             f'last100_eval_score: {mean_100_eval_score:5.1f}\u00B1{std_100_eval_score:5.1f}, '
                             f'last100_exploration: {mean_100_exp_rat:2.1f}\u00B1{std_100_exp_rat:2.1f}, '
                             f'replay_buffer: {len(self.replay_buffer)}'
                            )

        if mean_eval_reward >= self.max_eval_reward:
            self.max_eval_reward = mean_eval_reward

            self.agent1.save(self.checkpoints_dir, 'agent1')


def main():
    trainer = Trainer()
    for epoch in itertools.count():
        trainer.run_epoch(epoch)

        
if __name__ == '__main__':
    main()
