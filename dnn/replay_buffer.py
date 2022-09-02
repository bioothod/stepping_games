from bloom_filter2 import BloomFilter

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
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.experience_bloom_filter = BloomFilter(max_elements=100_000_000, error_rate=0.01)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def filled(self):
        return len(self) / self.capacity

    def add(self, obs, action, reward, next_obs, done):
        flat_obs = obs.reshape(-1)
        key = np.concatenate([flat_obs, [action]]).astype(np.uint8).tolist()
        if key in self.experience_bloom_filter:
            return

        self.experience_bloom_filter.add(key)

        np.copyto(self.states[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.as_tensor(self.states[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device).long()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device).float()
        dones = torch.as_tensor(self.dones[idxs], device=self.device)

        return states, actions, rewards, next_states, dones
