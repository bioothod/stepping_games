import itertools
import os

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

        self.batch_size = config.batch_size
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

    def forward_one(self, inputs):
        with torch.no_grad():
            states_features = self.state_features_model(inputs)

        v = self.values(states_features)
        v = v.squeeze(1)
        return v

    def forward(self, inputs):
        return_values = []

        start_index = 0
        while start_index < len(inputs):
            batch = inputs[start_index : start_index + self.batch_size, ...]
            ret = self.forward_one(batch)
            return_values.append(ret)

            start_index += len(batch)

        return_values = torch.cat(return_values, 0)
        return return_values

class Actor(nn.Module):
    def __init__(self, config, feature_model_creation_func):
        super().__init__()

        self.batch_size = config.batch_size

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

    def forward_one(self, inputs):
        if self.train_state_features:
            state_features = self.state_features(inputs)
        else:
            with torch.no_grad():
                state_features = self.state_features(inputs)

        outputs = self.features(state_features)
        return outputs

    def forward(self, inputs):
        return_logits = []

        start_index = 0
        while start_index < len(inputs):
            batch = inputs[start_index : start_index + self.batch_size, ...]
            ret = self.forward_one(batch)
            return_logits.append(ret)

            start_index += len(batch)

        return_logits = torch.cat(return_logits, 0)
        return return_logits

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

class PPO(train_selfplay.BaseTrainer):
    def __init__(self):
        self.config = edict({
            'player_ids': [1, 2],
            'train_player_id': 1,
            
            'load_checkpoints_dir': 'checkpoints_simple3_ppo_9',
            'checkpoints_dir': 'checkpoints_simple3_ppo_9',
            'eval_checkpoint_path': 'checkpoints_simple3_ppo_7/ppo_100.ckpt',

            'eval_after_train_steps': 20,

            'max_episode_len': 42,

            'policy_optimization_steps': 5,
            'policy_clip_range': 0.1,
            'policy_stopping_kl': 0.2,

            'value_optimization_steps': 5,
            'value_clip_range': float('inf'),
            'value_stopping_mse': 0.2,

            'states_sampling_ratio': 3,

            'entropy_loss_weight': 0.1,

            'gamma': 0.99,
            'tau': 0.97,

            'init_lr': 1e-4,

            'num_games_to_stop_training_state_model': 100_000_000,

            'num_features': 512,
            'hidden_dims': [128],

            'max_gradient_norm': 1.0,

            'num_training_games': 1024,

            'batch_size': 1024*14,
            'experience_buffer_to_batch_size_ratio': 2,
        })

        self.config.tensorboard_log_dir = os.path.join(self.config.checkpoints_dir, 'tensorboard_logs')

        #self.config.num_training_games = int(self.config.batch_size * self.config.experience_buffer_to_batch_size_ratio / len(self.config.player_ids))
        def align(x, alignment):
            return int((x + alignment - 1) / alignment) * alignment

        #self.config.num_training_games = align(self.config.num_training_games, 512)

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

        self.train_global_step = torch.zeros(1).long()

        if not os.path.exists(os.path.join(self.config.checkpoints_dir, 'train.log')):
            self.logger.info(f'actor:\n{print_networks("actor", self.actor, verbose=True)}')
            self.logger.info(f'critic:\n{print_networks("critic", self.critic, verbose=True)}')

        model_loaded = self.try_load(self.name, self)

        self.train_env = connectx_impl.ConnectX(self.config, self.critic, self.summary_writer, 'train', self.train_global_step)

        self.max_eval_metric = 0.0
        if model_loaded:
            eval_time_start = perf_counter()
            self.max_eval_metric, eval_rewards = self.evaluation.evaluate(self)
            self.max_mean_eval_metric = np.mean(eval_rewards)
            self.eval_global_step += 1
            eval_time = perf_counter() - eval_time_start

            self.logger.info(f'initial evaluation metric against {self.eval_agent_name}: {self.max_eval_metric:.2f}, mean: {self.max_mean_eval_metric:.4f} evaluation time: {eval_time:.1f} sec')

        #exit(0)

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
        min_total_loss = None
        total_sampled = 0
        for optimization_step in range(self.config.policy_optimization_steps):
            batch_indexes = np.random.choice(num_samples, min(num_samples, self.config.batch_size), replace=False)
            total_sampled += len(batch_indexes)

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

            total_loss = loss.item()
            if min_total_loss is None or total_loss < min_total_loss:
                min_total_loss = total_loss

                if optimization_step < self.config.policy_optimization_steps - 1 and total_sampled < len(states) * self.config.states_sampling_ratio:
                    continue

            with torch.no_grad():
                pred_log_probs_all, _ = self.actor.get_predictions(states, actions)
                kl = (log_probs - pred_log_probs_all).mean()
                kls.append(kl.item())
                if kl.item() > self.config.policy_stopping_kl:
                    break

            if total_sampled >= len(states):
                break

        mean_policy_loss = np.mean(policy_losses)
        mean_entropy_loss = np.mean(entropy_losses)
        mean_total_loss = np.mean(total_losses)
        mean_kl = np.mean(kls)

        self.summary_writer.add_scalars('train/actor_loss', {
            'policy': mean_policy_loss,
            'entropy': mean_entropy_loss,
            'total': mean_total_loss,
            'kl': mean_kl,
        }, self.train_global_step)

        self.summary_writer.add_scalar('train_iterations/actor', len(total_losses), self.train_global_step)
        self.summary_writer.add_scalar('train_iterations/samples', len(states), self.train_global_step)


        self.logger.debug(f'optimize_actor : '
                         f'iterations: {len(policy_losses)}/{self.config.policy_optimization_steps}, '
                         f'experiences: {len(states)}, '
                         f'total_loss: {mean_total_loss:.4f}, '
                         f'policy_loss: {mean_policy_loss:.4f}, '
                         f'entropy_loss: {mean_entropy_loss:.4f}, '
                         f'kl: {mean_kl:.3f}, '
                         f'stopping_kl: {self.config.policy_stopping_kl}')

    def optimize_critic(self, states, returns, values):
        num_samples = len(states)

        value_losses = []
        mse_values = []
        mse_returns = []
        min_total_loss = None
        total_sampled = 0

        for optimization_step in range(self.config.value_optimization_steps):
            batch_indexes = np.random.choice(num_samples, min(num_samples, self.config.batch_size), replace=False)
            total_sampled += len(batch_indexes)

            states_sampled = states[batch_indexes]
            returns_sampled = returns[batch_indexes]
            values_sampled = values[batch_indexes]

            pred_values = self.critic(states_sampled)
            pred_values_clipped = values_sampled + (pred_values - values_sampled).clamp(-self.config.value_clip_range, self.config.value_clip_range)

            v_loss = (returns_sampled - pred_values).pow(2)
            v_loss_clipped = (returns_sampled - pred_values_clipped).pow(2)
            value_loss = torch.max(v_loss, v_loss_clipped).mul(0.5).mean()

            self.critic_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_gradient_norm)
            self.critic_opt.step()

            value_loss = value_loss.item()
            value_losses.append(value_loss)

            if min_total_loss is None or value_loss < min_total_loss:
                min_total_loss = value_loss

                if optimization_step < self.config.value_optimization_steps - 1 and total_sampled < len(states) * self.config.states_sampling_ratio:
                    continue

            with torch.no_grad():
                pred_values_all = self.critic(states)

                mse_val = (values - pred_values_all).pow(2).mul(0.5).mean()
                mse_ret = (returns - pred_values_all).pow(2).mul(0.5).mean()

                mse_values.append(mse_val.item())
                mse_returns.append(mse_ret.item())

                if mse_val.item() > self.config.value_stopping_mse or mse_ret.item() > self.config.value_stopping_mse:
                    break

            if total_sampled >= len(states):
                break

        mean_value_loss = np.mean(value_losses)
        mean_mse_values = np.mean(mse_values)
        mean_mse_returns = np.mean(mse_returns)

        self.summary_writer.add_scalars('train/critic_loss', {
            'mse_value': mean_mse_values,
            'mse_returns': mean_mse_returns,
            'value': mean_value_loss,
        }, self.train_global_step)
        self.summary_writer.add_scalar('train_iterations/critic', len(value_losses), self.train_global_step)


        self.logger.debug(f'optimize_critic: '
                         f'iterations: {len(value_losses)}/{self.config.value_optimization_steps}, '
                         f'experiences: {len(states)}, '
                         f'mean: '
                         f'value_loss: {mean_value_loss:.4f}, '
                         f'mse_values: {mean_mse_values:.3e}, '
                         f'mse_returns: {mean_mse_returns:.4f}, '
                         f'stopping_mse: {self.config.value_stopping_mse}')

    def try_train(self):
        self.fill_episode_buffer()

        self.set_training_mode(True)

        game_index, states, actions, log_probs, gaes, values, returns = self.train_env.dump()

        self.logger.debug(f'dump: episode_buffers: '
                         f'completed_games: {len(game_index)}, '
                         f'experiences: {len(states)}, '
                         f'states: {states.shape}, '
                         f'actions: {actions.shape}, '
                         f'log_probs: {log_probs.shape}, '
                         f'values: {values.shape}, '
                         f'returns: {returns.shape}, '
                         f'gaes: {gaes.shape}')

        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        self.optimize_actor(states, actions, log_probs, gaes)
        self.optimize_critic(states, returns, values)

        self.train_global_step += 1

        return True

    def make_single_step(self, player_id, game_index, states):
        states = states.to(self.config.device)

        with torch.no_grad():
            self.actor.train(False)
            actions, log_probs, explorations = self.actor.dist_actions(states)
            self.actor.train(True)

        new_states, rewards, dones = self.train_env.step(player_id, game_index, actions)

        return edict({
            'player_id': player_id,
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'dones': dones,
            'explorations': explorations,
            'new_states': new_states,
        })

    def make_single_step_and_save(self, player_id):
        game_index, states = self.train_env.current_states(player_id)
        if len(states) == 0:
            #self.logger.info(f'player_id: {player_id}: states: {len(states)}')
            return

        episode_len = self.train_env.episode_len[game_index]

        step = self.make_single_step(player_id, game_index, states)

        truncated_indexes = torch.logical_and(episode_len + 1 == self.config.max_episode_len, step.dones != True)
        step.dones[truncated_indexes] = 1

        next_values = torch.zeros(len(step.states), dtype=torch.float32)
        if truncated_indexes.sum() > 0:
            with torch.no_grad():
                next_truncated_states = step.new_states[truncated_indexes, ...]
                next_truncated_states = next_truncated_states.to(self.config.device)
                self.critic.train(False)
                nv = self.critic(next_truncated_states).detach().cpu()
                self.critic.train(True)
                next_values[truncated_indexes] = nv

        self.train_env.update_game_rewards(player_id, game_index, step.states, step.actions, step.log_probs, step.rewards, step.dones, step.explorations, next_values)

    def make_step(self):
        self.make_single_step_and_save(1)
        self.make_single_step_and_save(2)


        if self.train_env.total_games_completed > self.config.num_games_to_stop_training_state_model:
            if self.actor.train_state_features:
                self.logger.info(f'completed_games: {self.train_env.total_games_completed}, '
                                 f'num_games_to_stop_training_state_model: {self.config.num_games_to_stop_training_state_model}: '
                                 f'finishing training the state model')

            self.actor.train_state_features = False

    def fill_episode_buffer(self):
        self.train_env.reset()

        requested_states_size = self.config.batch_size * self.config.experience_buffer_to_batch_size_ratio
        winning_rate = float(self.max_eval_metric) / 100.
        number_of_states = winning_rate * self.config.num_training_games * self.config.max_episode_len * len(self.config.player_ids)
        number_of_states_or_requested_states = max(number_of_states, requested_states_size)

        completed_games = 0
        completed_states = 0
        while len(self.train_env.running_index()) > 0:
            self.make_step()

            completed_games, completed_states = self.train_env.completed_games_and_states()

            self.logger.debug(f'fill_episode_buffer: '
                             f'running_games: {len(self.train_env.running_index())}, '
                             f'completed_games: {completed_games}, '
                             f'completed_states: {completed_states}, '
                             f'requested_size {requested_states_size}')

            if completed_states >= number_of_states_or_requested_states:
                break

        self.summary_writer.add_scalar('train_iterations/completed_games', completed_games, self.train_global_step)
        self.summary_writer.add_scalars('train_iterations/completed_states', {
            'completed_states': completed_states,
            'config_requested_states_size': requested_states_size,
            'winning_rate_number_of_states': number_of_states,
        }, self.train_global_step)


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
