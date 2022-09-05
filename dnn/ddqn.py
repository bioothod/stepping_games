import numpy as np
import torch
import torch.nn as nn

import replay_buffer

class DDQNModel(nn.Module):
    def __init__(self, config, feature_model_creation_func, logger):
        super().__init__()

        self.feature_model = feature_model_creation_func(config)

        hidden_dims = [config.num_features] + config.hidden_dims
        modules = []

        for i in range(1, len(hidden_dims)):
            input_dim = hidden_dims[i-1]
            output_dim = hidden_dims[i]

            l = nn.Linear(input_dim, output_dim)
            modules.append(l)
            modules.append(nn.ReLU(inplace=True))
        
        self.features = nn.Sequential(*modules)

        self.output_value = nn.Linear(hidden_dims[-1], 1)
        self.output_adv = nn.Linear(hidden_dims[-1], config.num_actions)

    def forward(self, inputs):
        low_level_features = self.feature_model(inputs)
        features = self.features(low_level_features)

        a = self.output_adv(features)
        v = self.output_value(features)
        v = v.expand_as(a)
        
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q

class DDQN:
    def __init__(self, name, config, feature_model_creation_func, logger):
        #super().__init__(name, config)
        self.logger = logger
        self.name = name
        self.config = config

        self.prev_experience = None
        self.replay_buffer = replay_buffer.ReplayBuffer(obs_shape=self.config.observation_shape,
                                                        obs_dtype=np.float32,
                                                        action_shape=(self.config.num_actions, ),
                                                        capacity=self.config.replay_buffer_size,
                                                        device=self.config.device)

        self.model = DDQNModel(config, feature_model_creation_func, logger).to(config.device)
        self.target_model = DDQNModel(config, feature_model_creation_func, logger).to(config.device)
        self.target_model.train(False)

        self.update_network(tau=1.0)

        self.max_gradient_norm = 1
        self.value_opt = torch.optim.Adam(self.model.parameters(), lr=config.init_lr)

    def set_training_mode(self, training):
        self.model.train(training)

    def train(self, batch):
        self.model.zero_grad()
        self.model.train()

        states, actions, rewards, next_states, is_terminals = batch
        batch_size = len(is_terminals)

        argmax_a_q_sp = self.model(next_states).max(1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + self.config.gamma * max_a_q_sp * (1 - is_terminals)

        q_sa = self.model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()

        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
        self.value_opt.step()

    def update_network(self, tau=None):
        tau = self.config.tau if tau is None else tau
        for target, online in zip(self.target_model.parameters(), self.model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data

            mixed_weights = target_ratio + online_ratio
            target.data = mixed_weights.detach().clone()

    def save(self, checkpoint_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.value_opt.state_dict(),
            }, checkpoint_path)

        return checkpoint_path

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.value_opt.load_state_dict(checkpoint['optimizer_state_dict'])

        self.logger.info(f'{self.name}: loaded checkpoint {checkpoint_path}')

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.config.device, dtype=torch.float32)
        return x

    def __call__(self, state):
        state = self._format(state)

        action = self.model(state)
        return action

    def try_train(self):
        if len(self.replay_buffer) < self.config.batch_size * self.config.num_warmup_batches:
            return False

        new_batch_size = self.config.batch_size + 1024
        if len(self.replay_buffer) >= new_batch_size and new_batch_size <= self.config.max_batch_size:
            self.logger.info(f'train: batch_size update: {self.config.batch_size} -> {new_batch_size}')
            self.config.batch_size = new_batch_size

        self.set_training_mode(True)
        experiences = self.replay_buffer.sample(self.config.batch_size)
        self.train(experiences)
        self.update_network()

        return True

    def add_flipped_state(self, state, action, reward, new_state, done):
        state_flipped = np.flip(state, 2)
        action_flipped = self.config.num_actions - action - 1
        new_state_flipped = np.flip(new_state, 2)
        self.replay_buffer.add(state_flipped, action_flipped, reward, new_state_flipped, done)

    def make_opposite(self, state):
        state_opposite = state.detach().clone()
        state_opposite[state == 1] = 2
        state_opposite[state == 2] = 1
        return state_opposite

    def add_experiences(self, states, actions, rewards, new_states, dones):
        for state, action, reward, new_state, done in zip(states, actions, rewards, new_states, dones):
            self.replay_buffer.add(state, action, reward, new_state, done)
            self.add_flipped_state(state, action, reward, new_state, done)

    def make_single_step_and_save(self, player_id):
        states = self.train_env.current_states()

        # agent's network assumes inputs are always related to the first player
        if player_id == 2:
            states = self.make_opposite(states)

        states = states.to(self.config.device)

        actions, explorations = self.train_action_strategy.select_action(self.train_agent, states)
        actions = torch.from_numpy(actions)
        new_states, rewards, dones = self.train_env.step(player_id, actions)

        if player_id == 2:
            new_states = self.make_opposite(new_states)


        states = states.detach().cpu().numpy()
        new_states = new_states.detach().cpu().numpy()

        if self.prev_experience is not None:
            prev_states, prev_actions, prev_rewards, prev_new_states, prev_dones = self.prev_experience

            cur_win_index = rewards == 1
            prev_rewards[cur_win_index] = -1
            prev_dones[cur_win_index] = 1

            #cur_lose_index = rewards < 0
            #prev_rewards[cur_lose_index] = 1
            #prev_dones[cur_lose_index] = 1

            self.add_experiences(prev_states, prev_actions, prev_rewards, prev_new_states, prev_dones)

        self.prev_experience = deepcopy((states, actions, rewards, new_states, dones))

        return rewards, dones, explorations
