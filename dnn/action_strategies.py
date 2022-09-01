import numpy as np
import torch

class GreedyStrategy():
    def __init__(self):
        pass
    
    def select_action(self, model, state):
        self.exploratory_action_taken = np.ones(len(state), dtype=np.int32)
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()

        actions = np.argmax(q_values, 1)
        exploratory_actions_taken = np.zeros_like(actions, dtype=np.int32)
        return actions, exploratory_actions_taken

class EGreedyStrategy():
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()

        batch_size, num_actions = q_values.shape[:2]

        policy_actions = np.argmax(q_values, 1)
        random_actions = np.random.randint(0, num_actions, batch_size)
        
        actions = np.where(np.random.rand(batch_size) > self.epsilon, policy_actions, random_actions)
        
        exploratory_action_taken = (actions != policy_actions).astype(np.int32)
        return actions, exploratory_action_taken

class EGreedyLinearStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, max_steps=20000):
        self.t = 0
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.max_steps = max_steps
        self.exploratory_action_taken = None
        
    def _epsilon_update(self, batch_size):
        epsilon = 1 - (np.arange(batch_size) + self.t) / self.max_steps
        epsilon = (self.init_epsilon - self.min_epsilon) * epsilon + self.min_epsilon
        epsilon = np.clip(epsilon, self.min_epsilon, self.init_epsilon)
        self.t += batch_size
        return epsilon

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()

        batch_size, num_actions = q_values.shape[:2]

        policy_actions = np.argmax(q_values, 1)
        random_actions = np.random.randint(0, num_actions, batch_size)
        
        epsilon = self._epsilon_update(batch_size)
        actions = np.where(np.random.rand(batch_size) > epsilon, policy_actions, random_actions)

        exploratory_action_taken = (actions != policy_actions).astype(np.int32)
        return actions, exploratory_action_taken

class EGreedyExpStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0

    def _epsilon_update(self, batch_size):
        if self.t < self.decay_steps:
            epsilon = self.epsilons[self.t : self.t + batch_size]
            if len(epsilon) < batch_size:
                epsilon_add = [self.min_epsilon] * (batch_size - len(epsilon))
                epsilon = np.concatenate([epsilon, epsilon_add], 0)
        else:
            epsilon = [self.min_epsilon] * batch_size

        self.t += batch_size
        return epsilon

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).detach().cpu().data.numpy()

        batch_size, num_actions = q_values.shape[:2]

        policy_actions = np.argmax(q_values, 1)
        random_actions = np.random.randint(low=0, high=num_actions, size=batch_size)

        epsilon = self._epsilon_update(batch_size)
        actions = np.where(np.random.rand(batch_size) > epsilon, policy_actions, random_actions)

        exploratory_action_taken = (actions != policy_actions).astype(np.int32)
        return actions, exploratory_action_taken

class SoftMaxStrategy():
    def __init__(self, 
                 init_temp=1.0, 
                 min_temp=0.3, 
                 exploration_ratio=0.8, 
                 max_steps=25000):
        self.t = 0
        self.init_temp = init_temp
        self.exploration_ratio = exploration_ratio
        self.min_temp = min_temp
        self.max_steps = max_steps
        
    def _update_temp(self, batch_size):
        temp = 1 - self.t / (self.max_steps * self.exploration_ratio)
        temp = (self.init_temp - self.min_temp) * temp + self.min_temp
        temp = np.clip(temp, self.min_temp, self.init_temp)
        self.t += 1
        return temp

    def select_action(self, model, state):
        batch_size = state.shape[0]
        temp = self._update_temp(batch_size)

        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()
            scaled_qs = q_values/temp
            norm_qs = scaled_qs - scaled_qs.max(1)
            e = np.exp(norm_qs)
            probs = e / np.sum(e)
            assert np.isclose(probs.sum(), 1.0)

        action = np.random.choice(np.arange(len(probs)), size=1, p=probs)[0]
        exploratory_action_taken = action != np.argmax(q_values)
        return action, exploratory_action_taken
