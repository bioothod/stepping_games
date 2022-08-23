import numpy as np

import torch

class ReplayBuffer:
    def __init__(self, obs_shape, obs_dtype, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        self.states = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_states = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.states[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.choice(range(0, len(self)), size=batch_size, replace=False)

        states = torch.as_tensor(self.states[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return states, actions, rewards, next_states, not_dones
