from collections import defaultdict
import itertools
import os

from easydict import EasyDict as edict
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

import connectx_impl
import networks
from print_networks import print_networks
from rl_agents import Actor, Critic
import train_selfplay


class PPO(train_selfplay.BaseTrainer):
    def __init__(self):
        self.config = edict({
            'player_ids': [1, 2],
            'train_player_id': 1,

            'load_checkpoints_dir': 'checkpoints_ppo_13_large_simple2_model',
            'checkpoints_dir': 'checkpoints_ppo_13_large_simple2_model',
            'eval_agent_template': 'submission/feature_model_ppo6.py:submission/rl_agents_ppo6.py:checkpoints_simple3_ppo_6/ppo_100.ckpt',
            'score_evaluation_dataset': 'refmoves1k_kaggle',

            'eval_after_train_steps': 20,

            'max_episode_len': 42,

            'policy_optimization_steps': 10,
            'policy_clip_range': 0.1,
            'policy_stopping_kl': 0.02,

            'value_optimization_steps': 10,
            'value_clip_range': float('inf'),
            'value_stopping_mse': 0.85,

            'entropy_loss_weight': 0.1,

            'gamma': 1.0,
            'tau': 0.99,
            'default_reward': 0,

            'init_lr': 1e-5,

            'num_games_to_stop_training_state_model': 100_000_000,

            'num_features': 512,
            'hidden_dims': [128],

            'max_gradient_norm': 1.0,

            'num_training_games': 1024*3,

            'batch_size': 1024*16,
            'experience_buffer_to_batch_size_ratio': 2,
        })

        self.config.tensorboard_log_dir = os.path.join(self.config.checkpoints_dir, 'tensorboard_logs')
        first_run = True
        if os.path.exists(self.config.tensorboard_log_dir) and len(os.listdir(self.config.tensorboard_log_dir)) > 0:
            first_run = False

        super().__init__(self.config)
        self.name = 'ppo'

        def feature_model_creation_func(config):
            model = networks.simple2_model.Model(config)
            return model


        self.actor = Actor(self.config, feature_model_creation_func).to(self.config.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.init_lr)

        self.critic = Critic(self.config, self.actor.state_features_model).to(self.config.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.init_lr)

        self.global_step = torch.zeros(1).long()

        if first_run:
            self.logger.info(f'actor:\n{print_networks("actor", self.actor, verbose=True)}')
            self.logger.info(f'critic:\n{print_networks("critic", self.critic, verbose=True)}')

        model_loaded = self.try_load(self.name, self)

        self.train_env = connectx_impl.ConnectX(self.config)

        self.max_eval_metric = 0.0
        if model_loaded:
            eval_time_start = perf_counter()
            self.max_eval_metric, eval_rewards = self.evaluation.evaluate(self)
            self.max_mean_eval_metric = np.mean(eval_rewards)
            self.eval_global_step += 1
            eval_time = perf_counter() - eval_time_start

            self.logger.info(f'initial evaluation metric against {self.eval_agent_name}: {self.max_eval_metric:.2f}, mean: {self.max_mean_eval_metric:.4f} evaluation time: {eval_time:.1f} sec')

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

    def optimize_actor(self, states, actions, log_probs, gaes, state_probs):
        num_samples = len(actions)

        policy_losses = []
        entropy_losses = []
        total_losses = []
        kls = []
        min_total_loss = None
        total_sampled = 0
        for optimization_step in range(self.config.policy_optimization_steps):
            batch_indexes = np.random.choice(num_samples, min(num_samples, self.config.batch_size), replace=False)
            #batch_indexes = np.random.choice(num_samples, min(num_samples, self.config.batch_size), replace=False, p=state_probs)
            total_sampled += len(batch_indexes)

            states_batch = states[batch_indexes]
            actions_batch = actions[batch_indexes]
            log_probs_batch = log_probs[batch_indexes]
            gaes_batch = gaes[batch_indexes]

            pred_log_probs, pred_entropies = self.actor.get_predictions_from_states(states_batch, actions_batch)
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
                pred_log_probs_all, _ = self.actor.get_predictions_from_states(states, actions)

                kl = (log_probs - pred_log_probs_all).mean()
                kls.append(kl.item())
                if kl.item() > self.config.policy_stopping_kl:
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
        }, self.global_step)

        self.summary_writer.add_scalar('train_iterations/actor', len(total_losses), self.global_step)

        self.logger.debug(f'optimize_actor : '
                         f'iterations: {len(policy_losses)}/{self.config.policy_optimization_steps}, '
                         f'experiences: {len(states)}, '
                         f'total_loss: {mean_total_loss:.4f}, '
                         f'policy_loss: {mean_policy_loss:.4f}, '
                         f'entropy_loss: {mean_entropy_loss:.4f}, '
                         f'kl: {mean_kl:.3f}, '
                         f'stopping_kl: {self.config.policy_stopping_kl}')

    def optimize_critic(self, states, returns, values, state_probs):
        num_samples = len(states)

        value_losses = []
        mse_values = []
        mse_returns = []
        min_total_loss = None
        total_sampled = 0

        for optimization_step in range(self.config.value_optimization_steps):
            batch_indexes = np.random.choice(num_samples, min(num_samples, self.config.batch_size), replace=False)
            #batch_indexes = np.random.choice(num_samples, min(num_samples, self.config.batch_size), replace=False, p=state_probs)
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

            with torch.no_grad():
                pred_values_all = self.critic(states)

                mse_val = (values - pred_values_all).pow(2).mul(0.5).mean()
                mse_ret = (returns - pred_values_all).pow(2).mul(0.5).mean()

                mse_values.append(mse_val.item())
                mse_returns.append(mse_ret.item())

                if mse_val.item() > self.config.value_stopping_mse or mse_ret.item() > self.config.value_stopping_mse:
                    break

        mean_value_loss = np.mean(value_losses)
        mean_mse_values = np.mean(mse_values)
        mean_mse_returns = np.mean(mse_returns)

        self.summary_writer.add_scalars('train/critic_loss', {
            'mse_value': mean_mse_values,
            'mse_returns': mean_mse_returns,
            'value': mean_value_loss,
        }, self.global_step)
        self.summary_writer.add_scalar('train_iterations/critic', len(value_losses), self.global_step)


        self.logger.debug(f'optimize_critic: '
                         f'iterations: {len(value_losses)}/{self.config.value_optimization_steps}, '
                         f'experiences: {len(states)}, '
                         f'mean: '
                         f'value_loss: {mean_value_loss:.4f}, '
                         f'mse_values: {mean_mse_values:.3e}, '
                         f'mse_returns: {mean_mse_returns:.4f}, '
                         f'stopping_mse: {self.config.value_stopping_mse}')

    def try_train(self):
        self.set_training_mode(False)
        self.fill_episode_buffer()

        game_index, states, actions, log_probs, gaes, values, returns = self.train_env.dump(self.actor, self.critic, self.summary_writer, 'train', self.global_step)

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

        if False:
            unique_state_actions = defaultdict(float)
            dist = []

            unique_state_action_index = []
            for i, (state, action) in enumerate(zip(states, actions)):
                state_key = tuple(state.cpu().numpy().flatten().tolist())
                action_key = tuple(action.cpu().numpy().flatten().tolist())

                state_action_key = state_key + action_key
                dist.append(state_action_key)

                if state_action_key not in unique_state_actions:
                    unique_state_action_index.append(i)

                unique_state_actions[state_action_key] += 1.

            dist = torch.tensor(dist).float()
            self.summary_writer.add_histogram('train/state_action', dist, self.global_step, bins=100)
            self.summary_writer.add_scalars('train_iterations/samples', {
                'samples': len(states),
                'unique_state_actions': len(unique_state_actions),
            }, self.global_step)

            index = torch.tensor(unique_state_action_index).long()
            states = states[index]
            actions = actions[index]
            log_probs = log_probs[index]
            gaes = gaes[index]
            values = values[index]
            returns = returns[index]

            state_probs = np.ones(len(states), dtype=np.float32)
            for i, (state, action) in enumerate(zip(states, actions)):
                state_key = tuple(state.cpu().numpy().flatten().tolist())
                action_key = tuple(action.cpu().numpy().flatten().tolist())

                state_action_key = state_key + action_key
                state_probs[i] = unique_state_actions[state_action_key]

            state_probs /= state_probs.sum()
        else:
            state_probs = np.ones(len(states), dtype=np.float32)

        self.set_training_mode(True)

        self.optimize_actor(states, actions, log_probs, gaes, state_probs)
        self.optimize_critic(states, returns, values, state_probs)

        self.global_step += 1

        return True

    def make_single_step_and_save(self, player_id):
        game_index, game_states = self.train_env.current_states()
        if len(game_index) == 0:
            return

        with torch.no_grad():
            actions, log_probs, explorations = self.actor.dist_actions(player_id, game_states)

        new_states, rewards, dones = self.train_env.step(player_id, game_index, actions)

        episode_len = self.train_env.episode_len[game_index]
        truncated_indexes = torch.logical_and(episode_len + 1 == self.config.max_episode_len, dones != True)
        dones[truncated_indexes] = 1

        next_values = torch.zeros(len(game_index), dtype=torch.float32)
        if False:
            if truncated_indexes.sum() > 0:
                with torch.no_grad():
                    next_truncated_states = new_states[truncated_indexes, ...]
                    next_truncated_states = next_truncated_states.to(self.config.device)
                    nv = self.critic(next_truncated_states).detach().cpu()
                    next_values[truncated_indexes] = nv

        self.train_env.update_game_rewards(player_id, game_index, game_states, actions, log_probs, rewards, dones, next_values, explorations)

    def make_step(self):
        for player_id in self.config.player_ids:
            self.make_single_step_and_save(player_id)

        if self.train_env.total_games_completed > self.config.num_games_to_stop_training_state_model:
            if self.actor.train_state_features:
                self.logger.info(f'completed_games: {self.train_env.total_games_completed}, '
                                 f'num_games_to_stop_training_state_model: {self.config.num_games_to_stop_training_state_model}: '
                                 f'finishing training the state model')

            self.actor.train_state_features = False

    def fill_episode_buffer(self):
        self.train_env.reset()

        requested_states_size = self.config.batch_size * self.config.experience_buffer_to_batch_size_ratio
        winning_rate = float(self.max_eval_metric) / 100. * 1.5
        number_of_states = winning_rate * self.config.num_training_games * self.config.max_episode_len
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

        self.summary_writer.add_scalar('train_iterations/completed_games', completed_games, self.global_step)
        self.summary_writer.add_scalars('train_iterations/completed_states', {
            'completed_states': completed_states,
            'config_requested_states_size': requested_states_size,
            'winning_rate_number_of_states': number_of_states,
        }, self.global_step)

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
