from typing import Union

import numpy as np
import torch

EPS = 1e-5

class PER:
    def __init__(self,
                 rank_based=False,
                 alpha=0.6,
                 beta0=0.1,
                 beta_rate=0.99992):
        self.td_errors = Union[type(None), torch.Tensor]
        self.rank_based = rank_based # if not rank_based, then proportional
        self.alpha = alpha # how much prioritization to use 0 is uniform (no priority), 1 is full priority
        self.beta = beta0 # bias correction 0 is no correction 1 is full correction
        self.beta0 = beta0 # beta0 is just beta's initial value
        self.beta_rate = beta_rate
        self.td_errors: torch.Tensor = None

    def update(self, idxs, td_errors):
        self.td_errors[idxs] = td_errors.abs()

        if self.rank_based:
            sorted_idx = torch.argsort(self.td_errors, descending=True)
            return sorted_idx

        return torch.arange(len(td_errors))

    def initial_store(self, td_errors):
        self.td_errors = td_errors.abs()

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate**-1)

    def sample(self, batch_size):
        self._update_beta()

        num_entries = len(self.td_errors)
        if num_entries <= batch_size:
            return torch.arange(num_entries)

        if self.rank_based:
            # td_errors are sorted in descending order, so the first entry has the largest error, hence the largest priority
            priorities = 1 / (torch.arange(num_entries) + 1)
        else: # proportional
            priorities = self.td_errors

        scaled_priorities = priorities**self.alpha
        probs = scaled_priorities / scaled_priorities.sum()

        weights = (num_entries * probs)**-self.beta
        normalized_weights = weights/weights.max()
        idxs = np.random.choice(num_entries, batch_size, replace=False, p=probs.cpu().numpy())
        return idxs

        samples_stacks = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, self.sample_index]).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        return idxs_stack, weights_stack, samples_stacks
