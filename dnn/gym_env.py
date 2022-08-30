import gym
import random

import numpy as np

import kaggle_environments as kaggle

class ConnectXGym(gym.Env):
    def __init__(self, config, pair):
        self.config = config
        
        self.env = kaggle.make('connectx', configuration=config, debug=True)

        self.pair = pair

        self.action_shape = (self.config.columns,)
        self.action_space = gym.spaces.Discrete(self.config.columns)
        self.observation_dtype = np.float32
        self.observation_shape = (1, self.config.rows, self.config.columns)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=self.observation_dtype)

        self.reward_range = (-10, 1)

        self.session = self.env.train(self.pair)

    def switch_agents(self):
        self.pair = self.pair[::-1]
        self.session = self.env.train(self.pair)

    def reset(self):
        if random.random() < 0.5:
            self.switch_agents()
            
        obs = self.session.reset()
        self.state = self.create_state(obs)
        return self.state

    def create_state1(self, obs):
        mark = obs['mark']

        orig_state = np.asarray(obs['board']).reshape(self.config.rows, self.config.columns)
        
        state = np.zeros(self.observation_shape, self.observation_dtype)
        state[0, orig_state == mark] = 1
        
        if mark == 2:
            state[1, orig_state == 1] = 1
        else:
            state[1, orig_state == 2] = 1

        return state

    def create_state(self, obs):
        orig_state = np.asarray(obs['board'], dtype=self.observation_dtype).reshape(self.observation_shape)
        state = orig_state.copy()
        if obs['mark'] == 2:
            state[orig_state == 2] = 1
            state[orig_state == 1] = 2

        return state
    
    def change_reward(self, old_reward, done):
        if old_reward == 1:
            return 1

        if done:
            return -1

        return 1 / float(self.config.rows * self.config.columns)

    def step(self, action):
        int_action = int(action)

        is_valid = self.state[:, 0, int_action] == 0

        if np.all(is_valid):
            obs, old_reward, done, _ = self.session.step(int_action)
            reward = self.change_reward(old_reward, done)
            self.state = self.create_state(obs)
        else:
            reward = -10
            done = True
            
        return self.state, reward, done, {}
