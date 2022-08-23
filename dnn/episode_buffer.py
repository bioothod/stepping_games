import gc
import time

import numpy as np
import torch

class EpisodeBuffer:
    def __init__(self, state_dim, gamma, tau, num_workers, max_episodes, max_episode_steps, device):
        if max_episodes < num_workers:
            raise ValueError(f'max_episodes: {max_episodes} must be more than num_workers: {num_workers}')

        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.num_workers = num_workers
        self.max_episodes = max_episodes
        self.max_episode_steps = max_episode_steps
        self.device = device

        self._truncated_fn = np.vectorize(lambda x: 'TimeLimit.truncated' in x and x['TimeLimit.truncated'])

        self.discounts = np.logspace(0, max_episode_steps+1, num=max_episode_steps+1, base=gamma, endpoint=False, dtype=np.float128)
        self.tau_discounts = np.logspace(0, max_episode_steps+1, num=max_episode_steps+1, base=gamma*tau, endpoint=False, dtype=np.float128)

        self.clear()

    def clear(self):
        episode_shape = (self.max_episodes, self.max_episode_steps)
        
        states_shape = np.concatenate((episode_shape, self.state_dim))
        self.states_mem = np.empty(shape=states_shape, dtype=np.float64)
        self.states_mem[:] = np.nan

        self.actions_mem = np.empty(shape=episode_shape, dtype=np.uint8)
        self.actions_mem[:] = np.nan
        

        self.returns_mem = np.empty(shape=episode_shape, dtype=np.float32)
        self.returns_mem[:] = np.nan
        
        self.gaes_mem = np.empty(shape=episode_shape, dtype=np.float32)
        self.gaes_mem[:] = np.nan
        
        self.logpas_mem = np.empty(shape=episode_shape, dtype=np.float32)
        self.logpas_mem[:] = np.nan

        self.episode_steps = np.zeros(shape=(self.max_episodes), dtype=np.uint16)
        self.episode_reward = np.zeros(shape=(self.max_episodes), dtype=np.float32)
        self.episode_exploration = np.zeros(shape=(self.max_episodes), dtype=np.float32)
        self.episode_seconds = np.zeros(shape=(self.max_episodes), dtype=np.float64)

        self.current_ep_idxs = np.arange(self.num_workers, dtype=np.uint16)
        gc.collect()

    def fill(self, envs, policy_model, value_model):
        states = envs.reset()

        worker_rewards = np.zeros(shape=(self.num_workers, self.max_episode_steps), dtype=np.float32)
        worker_exploratory = np.zeros(shape=(self.num_workers, self.max_episode_steps), dtype=np.bool)
        worker_steps = np.zeros(shape=(self.num_workers), dtype=np.uint16)
        worker_seconds = np.array([time.time(),] * self.num_workers, dtype=np.float64)

        buffer_full = False
        while not buffer_full:
            if len(self.episode_steps[self.episode_steps > 0]) >= self.max_episodes/2:
                break

            with torch.no_grad():
                actions, logpas, are_exploratory = policy_model.np_pass(states)
                values = value_model(states)

            
            next_states, rewards, terminals, infos = envs.step(actions)
            
            self.states_mem[self.current_ep_idxs, worker_steps] = states
            self.actions_mem[self.current_ep_idxs, worker_steps] = actions
            self.logpas_mem[self.current_ep_idxs, worker_steps] = logpas

            worker_exploratory[np.arange(self.num_workers), worker_steps] = are_exploratory
            worker_rewards[np.arange(self.num_workers), worker_steps] = rewards

            for w_idx in range(self.num_workers):
                if worker_steps[w_idx] + 1 == self.max_episode_steps:
                    terminals[w_idx] = 1
                    infos[w_idx]['TimeLimit.truncated'] = True

            states = next_states
            worker_steps += 1

            if terminals.sum() == 0:
                continue
            
            idx_terminals = np.flatnonzero(terminals)
            next_values = np.zeros(shape=(self.num_workers))
            truncated = self._truncated_fn(infos)
            if truncated.sum():
                idx_truncated = np.flatnonzero(truncated)
                with torch.no_grad():
                    truncated_next_values = value_model(next_states[idx_truncated]).cpu().numpy()
                    next_values[idx_truncated] = truncated_next_values

            new_states = envs.reset(ranks=idx_terminals)
            states[idx_terminals] = new_states

            for w_idx in range(self.num_workers):
                if w_idx not in idx_terminals:
                    continue

                e_idx = self.current_ep_idxs[w_idx]
                T = worker_steps[w_idx]
                self.episode_steps[e_idx] = T
                self.episode_reward[e_idx] = worker_rewards[w_idx, :T].sum()
                self.episode_exploration[e_idx] = worker_exploratory[w_idx, :T].mean()
                self.episode_seconds[e_idx] = time.time() - worker_seconds[w_idx]

                ep_rewards = np.concatenate((worker_rewards[w_idx, :T], [next_values[w_idx]]))
                ep_discounts = self.discounts[:T+1]
                ep_returns = np.array([np.sum(ep_discounts[:T+1-t] * ep_rewards[t:]) for t in range(T)])
                self.returns_mem[e_idx, :T] = ep_returns

                ep_states = self.states_mem[e_idx, :T]
                with torch.no_grad():
                    ep_values = value_model(ep_states)
                    next_values_tensor = torch.tensor([next_values[w_idx]], device=value_model.device, dtype=torch.float32)
                    ep_values = torch.cat((value_model(ep_states), next_values_tensor))

                np_ep_values = ep_values.view(-1).cpu().numpy()
                ep_tau_discounts = self.tau_discounts[:T]
                deltas = ep_rewards[:-1] + self.gamma * np_ep_values[1:] - np_ep_values[:-1]
                gaes = np.array([np.sum(self.tau_discounts[:T-t] * deltas[t:]) for t in range(T)])
                self.gaes_mem[e_idx, :T] = gaes

                worker_exploratory[w_idx, :] = 0
                worker_rewards[w_idx, :] = 0
                worker_steps[w_idx] = 0
                worker_seconds[w_idx] = time.time()

                new_ep_id = max(self.current_ep_idxs) + 1
                if new_ep_id >= self.max_episodes:
                    buffer_full = True
                    break

                self.current_ep_idxs[w_idx] = new_ep_id

        ep_idxs = self.episode_steps > 0
        ep_t = self.episode_steps[ep_idxs]

        self.states_mem = [row[:ep_t[i]] for i, row in enumerate(self.states_mem[ep_idxs])]
        self.states_mem = np.concatenate(self.states_mem)
        self.actions_mem = [row[:ep_t[i]] for i, row in enumerate(self.actions_mem[ep_idxs])]
        self.actions_mem = np.concatenate(self.actions_mem)
        self.returns_mem = [row[:ep_t[i]] for i, row in enumerate(self.returns_mem[ep_idxs])]
        self.returns_mem = torch.tensor(np.concatenate(self.returns_mem), device=value_model.device)
        self.gaes_mem = [row[:ep_t[i]] for i, row in enumerate(self.gaes_mem[ep_idxs])]
        self.gaes_mem = torch.tensor(np.concatenate(self.gaes_mem), device=value_model.device)
        self.logpas_mem = [row[:ep_t[i]] for i, row in enumerate(self.logpas_mem[ep_idxs])]
        self.logpas_mem = torch.tensor(np.concatenate(self.logpas_mem), device=value_model.device)

        ep_r = self.episode_reward[ep_idxs]
        ep_x = self.episode_exploration[ep_idxs]
        ep_s = self.episode_seconds[ep_idxs]
        return ep_t, ep_r, ep_x, ep_s

    def get_stacks(self):
        return (self.states_mem, self.actions_mem, self.returns_mem, self.gaes_mem, self.logpas_mem)

    def __len__(self):
        return self.episode_steps[self.episode_steps > 0].sum()
