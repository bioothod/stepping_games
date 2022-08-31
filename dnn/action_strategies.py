import numpy as np
import torch

class GreedyStrategy():
    def __init__(self):
        pass
    
    def select_action(self, model, state):
        self.exploratory_action_taken = np.ones(len(state), dtype=np.int32)
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()
            return np.argmax(q_values, 1)

class EGreedyStrategy():
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def select_action(self, model, state):
        self.exploratory_action_taken = np.ones(len(state), dtype=np.int32)
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()

        batch_size, num_actions = q_values.shape[:2]

        policy_actions = np.argmax(q_values, 1)
        random_actions = np.random.randint(0, num_actions, batch_size)
        
        actions = np.where(np.random.rand(batch_size) > self.epsilon, policy_actions, random_actions)
        
        self.exploratory_action_taken = (actions != policy_actions).astype(np.int32)
        return actions

class EGreedyLinearStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, max_steps=20000):
        self.t = 0
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.max_steps = max_steps
        self.exploratory_action_taken = None
        
    def _epsilon_update(self):
        epsilon = 1 - self.t / self.max_steps
        epsilon = (self.init_epsilon - self.min_epsilon) * epsilon + self.min_epsilon
        epsilon = np.clip(epsilon, self.min_epsilon, self.init_epsilon)
        self.t += 1
        return epsilon

    def select_action(self, model, state):
        self.exploratory_action_taken = np.ones(len(state), dtype=np.int32)
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()

        batch_size, num_actions = q_values.shape[:2]

        policy_actions = np.argmax(q_values, 1)
        random_actions = np.random.randint(0, num_actions, batch_size)
        
        actions = np.where(np.random.rand(batch_size) > self.epsilon, policy_actions, random_actions)

        self.epsilon = self._epsilon_update()
        self.exploratory_action_taken = (actions != policy_actions).astype(np.int32)
        return actions

class EGreedyExpStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, model, state):
        self.exploratory_action_taken = np.ones(len(state), dtype=np.int32)
        with torch.no_grad():
            q_values = model(state).detach().cpu().data.numpy()

        batch_size, num_actions = q_values.shape[:2]

        policy_actions = np.argmax(q_values, 1)
        random_actions = np.random.randint(low=0, high=num_actions, size=batch_size)
        
        actions = np.where(np.random.rand(batch_size) > self.epsilon, policy_actions, random_actions)

        self._epsilon_update()
        self.exploratory_action_taken = (actions != policy_actions).astype(np.int32)
        return actions

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
        self.exploratory_action_taken = None
        
    def _update_temp(self):
        temp = 1 - self.t / (self.max_steps * self.exploration_ratio)
        temp = (self.init_temp - self.min_temp) * temp + self.min_temp
        temp = np.clip(temp, self.min_temp, self.init_temp)
        self.t += 1
        return temp

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        temp = self._update_temp()

        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()
            scaled_qs = q_values/temp
            norm_qs = scaled_qs - scaled_qs.max(1)
            e = np.exp(norm_qs)
            probs = e / np.sum(e)
            assert np.isclose(probs.sum(), 1.0)

        action = np.random.choice(np.arange(len(probs)), size=1, p=probs)[0]
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action
