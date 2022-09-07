import itertools

from easydict import EasyDict as edict
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import connectx_impl
import networks
import replay_buffer
import train_selfplay


class Critic(nn.Module):
    def __init__(self, config, state_features_model):
        super().__init__()

        self.state_features_model = state_features_model

        hidden_dims = [config.num_features] + config.hidden_dims + [1]
        modules = []

        for i in range(1, len(hidden_dims)):
            input_dim = hidden_dims[i-1]
            output_dim = hidden_dims[i]

            l = nn.Linear(input_dim, output_dim)
            modules.append(l)
            modules.append(nn.ReLU(inplace=True))

        self.values = nn.Sequential(*modules)

    def forward(self, inputs):
        with torch.no_grad():
            states_features = self.state_features_model(inputs)

        v = self.values(states_features)
        v = v.squeeze(1)
        return v

class Actor(nn.Module):
    def __init__(self, config, feature_model_creation_func):
        super().__init__()

        self.state_features_model = feature_model_creation_func(config)

        hidden_dims = [config.num_features] + config.hidden_dims + [config.num_actions]
        modules = []

        for i in range(1, len(hidden_dims)):
            input_dim = hidden_dims[i-1]
            output_dim = hidden_dims[i]

            l = nn.Linear(input_dim, output_dim)
            modules.append(l)
            modules.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*modules)

    def state_features(self, inputs):
        state_features = self.state_features_model(inputs)
        return state_features

    def forward(self, inputs):
        state_features = self.state_features(inputs)
        outputs = self.features(state_features)
        return outputs

    def dist_actions(self, inputs):
        logits = self.forward(inputs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()

        log_prob = dist.log_prob(actions)

        is_exploratory = actions != torch.argmax(logits, axis=1)
        return actions, log_prob, is_exploratory

    def select_actions(self, states):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions

    def get_predictions(self, states, actions):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropies = dist.entropy()
        return log_prob, entropies

    def greedy_actions(self, states):
        logits = self.forward(states)
        actions = torch.argmax(logits)
        return actions

class SingleEpisodeBuffer:
    def __init__(self, config, critic):
        self.critic = critic

        self.device = config.device
        self.gamma = config.gamma
        self.tau = config.tau
        self.max_episode_len = config.max_episode_len

        self.states = []
        self.actions = []
        self.log_probs = []
        self.returns = None
        self.gaes = None

        self.rewards = []
        self.exploration = []

    def __len__(self):
        return len(self.rewards)

    def add(self, state, action, log_prob, reward, expl):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)

        self.rewards.append(reward)
        self.exploration.append(expl)

    def calculate_rewards(self, next_value):
        episode_reward = sum(self.rewards)
        episode_exploration = np.mean(self.exploration)

        episode_len = len(self.rewards)

        episode_rewards = np.array(self.rewards + [next_value], dtype=np.float32)
        discounts = np.logspace(0, self.max_episode_len+1, num=self.max_episode_len+1, base=self.gamma, endpoint=False, dtype=np.float32)
        discounts = discounts[:len(episode_rewards)]

        episode_returns = []
        for t in range(episode_len):
            ret = np.sum(discounts[:len(discounts)-t] * episode_rewards[t:])
            episode_returns.append(ret)
        self.returns = np.array(episode_returns, dtype=np.float32)

        self.states = np.array(self.states, dtype=np.float32)


        with torch.no_grad():
            states = torch.from_numpy(self.states).to(self.device)
            state_values = self.critic(states).detach().cpu().numpy()
            episode_values = np.concatenate([state_values, [next_value]])

        episode_values = episode_values.flatten()

        tau_discounts = np.logspace(0, self.max_episode_len+1, num=self.max_episode_len+1, base=self.gamma*self.tau, endpoint=False, dtype=np.float32)
        tau_discounts = tau_discounts[:episode_len]
        deltas = episode_rewards[:-1] + self.gamma * episode_values[1:] - episode_values[:-1]
        gaes = []
        for t in range(episode_len):
            ret = np.sum(tau_discounts[:len(tau_discounts)-t] * deltas[t:])
            gaes.append(ret)
        self.gaes = np.array(gaes, dtype=np.float32)

class EpisodeBuffers:
    def __init__(self, config, critic):
        self.critic = critic

        self.config = config

        self.reset()

    def reset(self):
        self.completed_games = []
        self.completed_experiences = 0
        self.current_games = {}

        for player_id in self.config.player_ids:
            self.current_games[player_id] = [SingleEpisodeBuffer(self.config, self.critic) for _ in range(self.config.train_num_games)]

    def __len__(self):
        return self.completed_experiences

    def add(self, player_id, states, actions, log_probs, rewards, dones, explorations, next_values):
        for game_id, game in enumerate(self.current_games[player_id]):
            state = states[game_id]
            action = actions[game_id]
            log_prob = log_probs[game_id]
            reward = rewards[game_id]
            done = dones[game_id]
            exploration = explorations[game_id]
            next_value = next_values[game_id]

            game.add(state, action, log_prob, reward, exploration)
            if done:
                game.calculate_rewards(next_value)
                self.current_games[player_id][game_id] = SingleEpisodeBuffer(self.config, self.critic)
                self.completed_games.append(game)
                self.completed_experiences += len(game)

    def dump(self):
        states = []
        actions = []
        log_probs = []
        returns = []
        gaes = []

        for game in self.completed_games:
            states.append(game.states)
            actions.append(game.actions)
            log_probs.append(game.log_probs)
            returns.append(game.returns)
            gaes.append(game.gaes)

        states = torch.from_numpy(np.concatenate(states)).to(self.config.device)
        actions = torch.from_numpy(np.concatenate(actions)).to(self.config.device)
        log_probs = torch.from_numpy(np.concatenate(log_probs)).to(self.config.device)
        returns = torch.from_numpy(np.concatenate(returns)).to(self.config.device)
        gaes = torch.from_numpy(np.concatenate(gaes)).to(self.config.device)

        return states, actions, log_probs, returns, gaes

class PPO(train_selfplay.BaseTrainer):
    def __init__(self):
        self.config = edict({
            'checkpoints_dir': 'checkpoints_simple3_ppo_1',

            'eval_after_train_steps': 20,

            'max_episode_len': 100,

            'policy_optimization_steps': 100,
            'policy_clip_range': 0.1,
            'policy_stopping_kl': 0.02,

            'value_optimization_steps': 100,
            'value_clip_range': float('inf'),
            'value_stopping_mse': 25,

            'entropy_loss_weight': 0.01,

            'gamma': 0.99,
            'tau': 0.97,

            'train_num_games': 1024,
            'init_lr': 1e-4,

            'num_features': 512,
            'hidden_dims': [128],

            'max_gradient_norm': 1.0,

            'batch_size': 1024,
            'max_batch_size': 1024*32,
        })
        super().__init__(self.config)

        self.name = 'ppo'

        def feature_model_creation_func(config):
            model = networks.simple3_model.Model(config)
            #model = networks.conv_model.Model(config)
            return model


        self.actor = Actor(self.config, feature_model_creation_func).to(self.config.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.init_lr)

        self.critic = Critic(self.config, self.actor.state_features_model).to(self.config.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.init_lr)

        self.episode_lengths = np.zeros(self.config.train_num_games)

        self.prev_experience = None
        self.episode_buffers = EpisodeBuffers(self.config, self.critic)

        model_loaded = self.try_load(self.name, self)

        self.train_env = connectx_impl.ConnectX(self.config, self.config.train_num_games, replay_buffer=None)

        self.max_eval_metric = 0.0
        if model_loaded:
            eval_time_start = perf_counter()
            self.max_eval_metric, _ = self.evaluate(self)
            eval_time = perf_counter() - eval_time_start

            self.logger.info(f'initial evaluation metric: {self.max_eval_metric:.2f}, evaluation time: {eval_time:.1f} sec')


    def set_training_mode(self, training):
        self.actor.train(training)
        self.critic.train(training)

    def save(self, checkpoint_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_opt.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_opt.state_dict(),
            }, checkpoint_path)

        return checkpoint_path

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        self.logger.info(f'{self.name}: loaded checkpoint {checkpoint_path}')

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.config.device, dtype=torch.float32)
        return x

    def __call__(self, states):
        states = torch.from_numpy(states).to(self.config.device)
        actions = self.actor.select_actions(states)
        actions = F.one_hot(actions, self.config.num_actions)
        return actions

    def optimize_actor(self, states, actions, log_probs, gaes):
        num_samples = len(actions)

        policy_losses = []
        entropy_losses = []
        total_losses = []
        kls = []
        for _ in range(self.config.policy_optimization_steps):
            batch_indexes = np.random.choice(num_samples, self.config.batch_size, replace=False)

            states_batch = states[batch_indexes]
            actions_batch = actions[batch_indexes]
            log_probs_batch = log_probs[batch_indexes]
            gaes_batch = gaes[batch_indexes]

            pred_log_probs, pred_entropies = self.actor.get_predictions(states_batch, actions_batch)
            ratios = (pred_log_probs - log_probs_batch).exp()
            pi_obj = gaes_batch * ratios
            pi_obj_clipped = gaes_batch * ratios.clamp(1 - self.config.policy_clip_range, 1 + self.config.policy_clip_range)

            policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean()
            entropy_loss = -pred_entropies.mean() * self.config.entropy_loss_weight

            self.actor_opt.zero_grad()
            loss = policy_loss + entropy_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_gradient_norm)
            self.actor_opt.step()

            policy_losses.append(policy_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(loss.item())

            with torch.no_grad():
                pred_log_probs_all, _ = self.actor.get_predictions(states, actions)
                kl = (log_probs - pred_log_probs_all).mean()
                kls.append(kl.item())
                if kl.item() > self.config.policy_stopping_kl:
                    break

        mean_policy_loss = np.mean(policy_losses)
        mean_entropy_loss = np.mean(entropy_losses)
        mean_total_loss = np.mean(total_losses)
        mean_kl = np.mean(kls)
        self.logger.debug(f'optimize_actor: '
                         f'experiences: {len(actions)}, '
                         f'iterations: {len(policy_losses)}/{self.config.policy_optimization_steps}, '
                         f'total_loss: {mean_total_loss:.4f}, '
                         f'policy_loss: {mean_policy_loss:.4f}, '
                         f'entropy_loss: {mean_entropy_loss:.4f}, '
                         f'kl: {mean_kl:.3f}, '
                         f'stopping_kl: {self.config.policy_stopping_kl}')

    def optimize_critic(self, states, returns, values):
        num_samples = len(states)

        value_losses = []
        mses = []
        for _ in range(self.config.value_optimization_steps):
            batch_indexes = np.random.choice(num_samples, self.config.batch_size, replace=False)

            states_batch = states[batch_indexes]
            returns_batch = returns[batch_indexes]
            values_batch = values[batch_indexes]

            pred_values = self.critic(states_batch)
            pred_values_clipped = values_batch + (pred_values - values_batch).clamp(-self.config.value_clip_range, self.config.value_clip_range)

            v_loss = (returns_batch - pred_values).pow(2)
            v_loss_clipped = (returns_batch - pred_values_clipped).pow(2)
            value_loss = torch.max(v_loss, v_loss_clipped).mul(0.5).mean()

            self.critic_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_gradient_norm)
            self.critic_opt.step()

            value_losses.append(value_loss.item())
            with torch.no_grad():
                pred_values_all = self.critic(states)
                mse = (values - pred_values_all).pow(2).mul(0.5).mean()
                mses.append(mse.item())

                if mse.item() > self.config.value_stopping_mse:
                    break

        mean_value_loss = np.mean(value_losses)
        mean_mse = np.mean(mses)
        self.logger.debug(f'optimize_critic: '
                         f'iterations: {len(value_losses)}/{self.config.value_optimization_steps}, '
                         f'value_loss: {mean_value_loss:.4f}, '
                         f'mse: {mean_mse:.3f}, '
                         f'stopping_mse: {self.config.value_stopping_mse}')

    def try_train(self):
        self.fill_episode_buffer()

        self.set_training_mode(True)

        states, actions, log_probs, returns, gaes = self.episode_buffers.dump()
        self.logger.debug(f'dump: episode_buffers: '
                         f'completed_games: {len(self.episode_buffers.completed_games)}, '
                         f'experiences: {len(self.episode_buffers)}, '
                         f'states: {states.shape}, '
                         f'actions: {actions.shape}, '
                         f'log_probs: {log_probs.shape}, '
                         f'returns: {returns.shape}, '
                         f'gaes: {gaes.shape}')

        values = self.critic(states).detach()
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        self.optimize_actor(states, actions, log_probs, gaes)
        self.optimize_critic(states, returns, values)

        new_batch_size = self.config.batch_size + 1024
        if new_batch_size <= self.config.max_batch_size:
            self.logger.info(f'train: batch_size update: {self.config.batch_size} -> {new_batch_size}')
            self.config.batch_size = new_batch_size

        return True

    def add_flipped_states(self, player_id, states, actions, log_probs, rewards, dones, explorations, next_values):
        states_flipped = np.flip(states, 3)
        actions_flipped = self.config.num_actions - actions - 1

        self.episode_buffers.add(player_id, states_flipped, actions_flipped, log_probs, rewards, dones, explorations, next_values)

    def make_opposite(self, state):
        state_opposite = state.detach().clone()
        state_opposite[state == 1] = 2
        state_opposite[state == 2] = 1
        return state_opposite

    def make_single_step_and_save(self, player_id):
        states = self.train_env.current_states()

        # agent's network assumes inputs are always related to the first player
        if player_id == 2:
            states = self.make_opposite(states)

        states = states.to(self.config.device)

        with torch.no_grad():
            actions, log_probs, explorations = self.actor.dist_actions(states)

        new_states, rewards, dones = self.train_env.step(player_id, actions)

        if player_id == 2:
            new_states = self.make_opposite(new_states)

        truncated_indexes = np.flatnonzero(self.episode_lengths + 1 == self.config.max_episode_len)
        dones[truncated_indexes] = 1

        next_values = np.zeros(len(states), dtype=np.float32)
        if len(truncated_indexes) > 0:
            with torch.no_grad():
                next_truncated_states = new_states[truncated_indexes, ...]
                next_truncated_states = next_truncated_states.to(self.config.device)
                nv = self.critic(next_truncated_states).detach().cpu().numpy()
                next_values[truncated_indexes] = nv

        states = states.detach().cpu().numpy()
        new_states = new_states.detach().cpu().numpy()

        actions = actions.detach().cpu().numpy()
        log_probs = log_probs.detach().cpu().numpy()
        explorations = explorations.detach().cpu().numpy()

        if self.prev_experience is not None:
            prev_player_id, prev_states, prev_actions, prev_log_probs, prev_rewards, prev_dones, prev_explorations, prev_next_values = self.prev_experience

            cur_win_index = rewards == 1
            prev_rewards[cur_win_index] = -1
            prev_dones[cur_win_index] = 1

            #cur_lose_index = rewards < 0
            #prev_rewards[cur_lose_index] = 1
            #prev_dones[cur_lose_index] = 1

            self.episode_buffers.add(prev_player_id, prev_states, prev_actions, prev_log_probs, prev_rewards, prev_dones, prev_explorations, prev_next_values)
            #self.add_flipped_states(prev_player_id, prev_states, prev_actions, prev_log_probs, prev_rewards, prev_dones, prev_explorations, prev_next_values)

        self.prev_experience = deepcopy((player_id, states, actions, log_probs, rewards, dones, explorations, next_values))
        self.train_env.update_game_rewards(player_id, rewards, dones, explorations)

    def make_step(self):
        self.make_single_step_and_save(1)
        self.make_single_step_and_save(2)
        self.episode_lengths += 1

    def fill_episode_buffer(self):
        self.train_env.reset()
        self.episode_buffers.reset()
        self.episode_lengths *= 0

        while len(self.episode_buffers) < self.config.batch_size:
            #self.logger.info(f'fill_episode_buffer: completed_games: {len(self.episode_buffers.completed_games)}, experiences: {len(self.episode_buffers)}, batch_size: {self.config.batch_size}')
            self.make_step()

def main():
    ppo = PPO()
    for epoch in itertools.count():
        try:
            ppo.run_epoch(ppo.train_env, ppo)
        except Exception as e:
            ppo.logger.critical(f'type: {type(e)}, exception: {e}')
            ppo.stop()

            if type(e) != KeyboardInterrupt:
                raise


if __name__ == '__main__':
    main()