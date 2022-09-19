import itertools

from easydict import EasyDict as edict
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import connectx_impl
import networks
from print_networks import print_networks
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

        self.train_state_features = True
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
        if self.train_state_features:
            state_features = self.state_features(inputs)
        else:
            with torch.no_grad():
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
        actions = torch.argmax(logits, 1)
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

    def add(self, exp):
        for game_id, game in enumerate(self.current_games[exp.player_id]):
            state = exp.states[game_id]
            action = exp.actions[game_id]
            log_prob = exp.log_probs[game_id]
            reward = exp.rewards[game_id]
            done = exp.dones[game_id]
            exploration = exp.explorations[game_id]
            next_value = exp.next_values[game_id]

            game.add(state, action, log_prob, reward, exploration)
            if done:
                game.calculate_rewards(next_value)
                self.current_games[exp.player_id][game_id] = SingleEpisodeBuffer(self.config, self.critic)
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
            'load_checkpoints_dir': 'checkpoints_simple3_ppo_7',
            'checkpoints_dir': 'checkpoints_simple3_ppo_7',
            'eval_checkpoint_path1': 'checkpoints_simple3_ppo_6/ppo_100.ckpt',

            'eval_after_train_steps': 20,

            'max_episode_len': 42,

            'policy_optimization_steps': 10,
            'policy_clip_range': 0.1,
            'policy_stopping_kl': 0.02,

            'value_optimization_steps': 10,
            'value_clip_range': float('inf'),
            'value_stopping_mse': 0.01,

            'entropy_loss_weight': 0.1,

            'gamma': 0.99,
            'tau': 0.97,

            'train_num_games': 1024*2,
            'init_lr': 1e-5,

            'num_games_to_stop_training_state_model': 10_000_000,

            'num_features': 512,
            'hidden_dims': [128],

            'max_gradient_norm': 1.0,

            'batch_size': 1024,
            'max_batch_size': 1024*8,
            'experience_buffer_to_batch_size_ratio': 2,
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

        self.logger.info(f'actor:\n{print_networks("actor", self.actor, verbose=True)}')
        self.logger.info(f'critic:\n{print_networks("critic", self.critic, verbose=True)}')

        self.prev_experience = None
        self.episode_buffers = EpisodeBuffers(self.config, self.critic)

        model_loaded = self.try_load(self.name, self)

        self.train_env = connectx_impl.ConnectX(self.config, self.config.train_num_games)

        self.max_eval_metric = 0.0
        if model_loaded:
            eval_time_start = perf_counter()
            self.max_eval_metric, _ = self.evaluate(self)
            eval_time = perf_counter() - eval_time_start

            self.logger.info(f'initial evaluation metric against {self.eval_agent_name}: {self.max_eval_metric:.2f}, evaluation time: {eval_time:.1f} sec')


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
        actions = self.actor.greedy_actions(states)
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

    def make_state(self, player_id, game_states):
        num_games = len(game_states)

        states = torch.zeros((1 + len(self.config.player_ids), num_games, self.config.rows, self.config.columns), dtype=torch.float32)
        states[0, ...] = player_id

        for idx, pid in enumerate(self.config.player_ids):
            player_idx = game_states[:, 0, ...] == pid
            states[idx + 1, player_idx] = 1

        states = states.transpose(1, 0)
        return states

    def make_single_step_and_save(self, player_id):
        states = self.train_env.current_states()
        states = self.make_state(player_id, states)
        states = states.to(self.config.device)

        with torch.no_grad():
            actions, log_probs, explorations = self.actor.dist_actions(states)

        new_states, rewards, dones = self.train_env.step(player_id, actions)
        new_states = self.make_state(player_id, new_states)

        dones = dones.detach().cpu().numpy()
        rewards = rewards.detach().clone().cpu().numpy()

        truncated_indexes = np.flatnonzero(np.logical_and(self.train_env.episode_lengths + 1 == self.config.max_episode_len, dones != 1))
        dones[truncated_indexes] = 1

        return edict({
            'player_id': player_id,
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'dones': dones,
            'explorations': explorations,
            'new_states': new_states,
            'truncated_indexes': truncated_indexes,
        })

    def make_single_step_and_save(self, player_id):
        states = self.train_env.current_states()

        step = self.make_single_step(player_id, states)

        next_values = np.zeros(len(step.states), dtype=np.float32)
        if len(step.truncated_indexes) > 0:
            with torch.no_grad():
                next_truncated_states = step.new_states[truncated_indexes, ...]
                next_truncated_states = next_truncated_states.to(self.config.device)
                nv = self.critic(next_truncated_states).detach().cpu().numpy()
                next_values[step.truncated_indexes] = nv

        states = step.states.detach().clone().cpu().numpy()

        actions = step.actions.detach().clone().cpu().numpy()
        log_probs = step.log_probs.detach().clone().cpu().numpy()
        explorations = step.explorations.detach().clone().cpu().numpy()

        if self.prev_experience is not None:
            self.prev_experience.dones[step.dones == 1] = 1

            cur_win_index = step.rewards == 1
            self.prev_experience.rewards[cur_win_index] = -1
            self.prev_experience.dones[cur_win_index] = 1

            #cur_lose_index = rewards < 0
            #prev_rewards[cur_lose_index] = 1
            #prev_dones[cur_lose_index] = 1

            self.episode_buffers.add(self.prev_experience)

        self.prev_experience = edict({
            'player_id': player_id,
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': step.rewards,
            'dones': step.dones,
            'explorations': explorations,
            'next_values': next_values
        })

        self.train_env.update_game_rewards(player_id, step.rewards, step.dones, explorations)

    def make_step(self):
        self.make_single_step_and_save(1)
        self.make_single_step_and_save(2)

        if len(self.train_env.completed_games) > self.config.num_games_to_stop_training_state_model:
            if self.actor.train_state_features:
                self.logger.info(f'completed_games: {len(self.train_env.completed_games)}, '
                                 f'num_games_to_stop_training_state_model: {self.config.num_games_to_stop_training_state_model}: '
                                 f'finishing training the state model')

            self.actor.train_state_features = False

    def fill_episode_buffer(self):
        self.train_env.reset()
        self.episode_buffers.reset()

        while len(self.episode_buffers) < self.config.batch_size * self.config.experience_buffer_to_batch_size_ratio:
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
