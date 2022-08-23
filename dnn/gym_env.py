import gym
import random

from easydict import EasyDict as edict

import numpy as np

import kaggle_environments as kaggle

class ConnectXGym(gym.Env):
    def __init__(self, config, agent1, agent2):
        self.config = config
        
        self.env = kaggle.make('connectx', configuration=config, debug=True)

        self.pair = [agent1, agent2]

        self.action_shape = (self.config.columns,)
        self.action_space = gym.spaces.Discrete(self.config.columns)
        self.observation_dtype = np.float32
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(self.config.rows, self.columns, 1), dtype=self.observation_dtype)

        self.reward_range = (-10, 1)

        self.session = self.env.train(self.pair)

    def swtich_agents(self):
        self.pair = self.pair[::-1]
        self.session = self.env.train(self.pair)

    def reset(self):
        if random.random() < 0.5:
            self.switch_agents()
            
        obs = self.session.reset()
        self.state = self.create_state(obs)
        return self.state

    def create_state(self, obs):
        state = np.asarray(obs['board'])
        state = state.reshape(self.config.rows, self.columns, 1)
        state = state.astype(np.float32)
        return state
    
    def change_reward(self, old_reward, done):
        if old_reward == 1:
            return 1

        if done:
            return -1

        return 1 / float(self.config.rows * self.config.columns)

    def step(self, action):
        int_action = int(action)

        is_valid = self.state[:, int_action, :] == 0

        if is_valid:
            obs, old_reward, done, _ = self.session.step(int_action)
            reward = self.change_reward(old_reward, done)
            self.state = self.create_state(obs)
        else:
            reward = -10
            done = True
            
        return self.state, reward, done, {}
