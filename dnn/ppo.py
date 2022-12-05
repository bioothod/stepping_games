from typing import Optional

from collections import defaultdict
import itertools
import os

from copy import deepcopy
from easydict import EasyDict as edict
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

import evaluate
import evaluate_score
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

            'checkpoints_dir': 'checkpoints_ppo_29',
            'eval_agent_template': 'submission/feature_model_ppo6.py:submission/rl_agents_ppo6.py:checkpoints_simple3_ppo_6/ppo_100.ckpt',
            'score_evaluation_dataset': 'refmoves1k_kaggle',

            'eval_after_train_steps': 5,

            'max_episode_len': 42,

            'policy_optimization_steps': 10,
            'policy_clip_range': 0.1,
            'policy_stopping_kl': 0.02,

            'value_optimization_steps': 10,
            'value_clip_range': float('inf'),
            'value_stopping_mse': 0.85,

            'enable_reference_move_training': True,

            'entropy_loss_weight': 0.1,

            'weight_decay': 1e-3,

            'gamma': 1.0,
            'tau': 0.97,
            'default_reward': 0,

            'init_lr': 1e-5,

            'num_features': 512,
            'hidden_dims': [128],

            'max_gradient_norm': 1.0,

            'num_training_games': 1024*3,

            'batch_size': 1024*32,
            'experience_buffer_to_batch_size_ratio': 2,
            'train_break_after_num_sampled': 1,
        })

        self.config.tensorboard_log_dir = os.path.join(self.config.checkpoints_dir, 'tensorboard_logs')
        first_run = True
        if os.path.exists(self.config.tensorboard_log_dir) and len(os.listdir(self.config.tensorboard_log_dir)) > 0:
            first_run = False

        self.global_step = torch.zeros(1).long()

        super().__init__(self.config)
        self.name = 'ppo'

        def feature_model_creation_func(config):
            model = networks.simple4_model.Model(config)
            return model


        self.actor = Actor(self.config, feature_model_creation_func).to(self.config.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.init_lr)

        self.critic = Critic(self.config, self.actor.state_features_model).to(self.config.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.init_lr)

        if self.config.enable_reference_move_training:
            self.actor_ref_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.init_lr)
            self.ref_loss_ce = torch.nn.CrossEntropyLoss()
            self.ref_best_moves = torch.zeros(len(self.evaluation.score_eval_ds.best_moves), self.config.num_actions, dtype=torch.float32, device=self.config.device)
            self.ref_good_moves = torch.zeros(len(self.evaluation.score_eval_ds.good_moves), self.config.num_actions, dtype=torch.float32, device=self.config.device)
            for idx, (best_moves, good_moves) in enumerate(zip(self.evaluation.score_eval_ds.good_moves, self.evaluation.score_eval_ds.best_moves)):
                self.ref_best_moves[idx, best_moves] = 1. / len(best_moves)
                self.ref_good_moves[idx, good_moves] = 1. / len(good_moves)

            self.ref_game_states: Optional[torch.Tensor] = None
            for i, player_id in enumerate(self.config.player_ids):
                player_idx = torch.arange(len(self.ref_good_moves))[self.evaluation.score_eval_ds.game_player_ids == player_id]
                game_states = self.evaluation.score_eval_ds.game_states[player_idx]
                states = self.actor.create_state(player_id, game_states)
                if i == 0:
                    self.ref_game_states = torch.zeros([len(self.ref_good_moves)] + list(states.shape[1:]), dtype=states.dtype, device=self.config.device)

                self.ref_game_states[player_idx] = states.to(self.config.device)


        if first_run:
            self.logger.info(f'actor:\n{print_networks("actor", self.actor, verbose=True)}')
            self.logger.info(f'critic:\n{print_networks("critic", self.critic, verbose=True)}')

        self.train_env = connectx_impl.ConnectX(self.config)

        model_loaded = self.try_load(self.name, self)

        if model_loaded:
            eval_time_start = perf_counter()
            self.max_eval_metric, eval_rewards = self.evaluation.evaluate(self)
            eval_time = perf_counter() - eval_time_start

            self.logger.info(f'initial evaluation metric against {self.eval_agent_name}: '
                             f'max_eval_metric: {self.max_eval_metric:.2f}, '
                             f'mean: {self.max_mean_eval_metric:.4f}, '
                             f'max_score_metric: {self.max_score_metric:.1f}, '
                             f'evaluation time: {eval_time:.1f} sec')

    def set_training_mode(self, training):
        self.actor.train(training)
        self.critic.train(training)

    def save(self, checkpoint_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_opt.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_opt.state_dict(),
            'global_step': self.global_step,
            'total_games_completed': self.train_env.total_games_completed,
            'max_eval_metric': self.max_eval_metric,
            'max_mean_eval_metric': self.max_mean_eval_metric,
            'max_score_metric': self.max_score_metric,
            }, checkpoint_path)

        return checkpoint_path

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        if 'global_step' in checkpoint:
            self.global_step *= 0
            self.global_step += checkpoint['global_step']

            self.train_env.total_games_completed = checkpoint['total_games_completed']

            self.max_eval_metric = checkpoint['max_eval_metric']
            self.max_mean_eval_metric = checkpoint['max_mean_eval_metric']
            self.max_score_metric = checkpoint['max_score_metric']

        self.logger.info(f'{self.name}: loaded checkpoint {checkpoint_path}')

    def optimize_actor_reference_moves(self):
        self.actor_ref_opt.zero_grad()

        pred_actions = self.actor.forward_from_states(self.ref_game_states)

        best_actions_loss = self.ref_loss_ce(pred_actions, self.ref_best_moves)
        good_actions_loss = self.ref_loss_ce(pred_actions, self.ref_good_moves)

        loss = best_actions_loss + good_actions_loss
        loss.backward()
        self.actor_ref_opt.step()

        self.summary_writer.add_scalars('train/ref_moves', {
                'good': good_actions_loss.item(),
                'best': best_actions_loss.item(),
                'total': loss.item(),
        }, self.global_step)


    def optimize_actor(self, states, actions, log_probs, gaes, state_probs):
        num_samples = len(actions)

        policy_losses = []
        entropy_losses = []
        total_losses = []
        kls = []
        total_sampled = 0
        for optimization_step in range(self.config.policy_optimization_steps):
            batch_indexes = np.random.choice(num_samples, min(num_samples, self.config.batch_size), replace=False, p=state_probs)
            total_sampled += len(batch_indexes)

            states_batch = states[batch_indexes]
            actions_batch = actions[batch_indexes]
            log_probs_batch = log_probs[batch_indexes]
            gaes_batch = gaes[batch_indexes]

            self.actor_opt.zero_grad()
            pred_log_probs, pred_entropies = self.actor.get_predictions_from_states(states_batch, actions_batch)
            ratios = (pred_log_probs - log_probs_batch).exp()
            pi_obj = gaes_batch * ratios
            pi_obj_clipped = gaes_batch * ratios.clamp(1 - self.config.policy_clip_range, 1 + self.config.policy_clip_range)

            policy_loss = -torch.min(pi_obj, pi_obj_clipped)
            entropy_loss = -pred_entropies * self.config.entropy_loss_weight

            policy_loss = policy_loss.mean()
            entropy_loss = entropy_loss.mean()

            total_loss = policy_loss + entropy_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_gradient_norm)
            self.actor_opt.step()

            policy_losses.append(policy_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(total_loss.item())

            with torch.no_grad():
                pred_log_probs_all, _ = self.actor.get_predictions_from_states(states, actions)

                kl = (log_probs - pred_log_probs_all).mean()
                kls.append(kl.item())
                if kl.item() > self.config.policy_stopping_kl:
                    if total_sampled > self.config.train_break_after_num_sampled * num_samples:
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
        total_sampled = 0

        for optimization_step in range(self.config.value_optimization_steps):
            batch_indexes = np.random.choice(num_samples, min(num_samples, self.config.batch_size), replace=False, p=state_probs)
            total_sampled += len(batch_indexes)

            states_batch = states[batch_indexes]
            returns_batch = returns[batch_indexes]
            values_batch = values[batch_indexes]

            self.critic_opt.zero_grad()

            pred_values = self.critic(states_batch)
            pred_values_clipped = values_batch + (pred_values - values_batch).clamp(-self.config.value_clip_range, self.config.value_clip_range)

            v_loss = (returns_batch - pred_values).pow(2)
            v_loss_clipped = (returns_batch - pred_values_clipped).pow(2)
            value_loss = torch.max(v_loss, v_loss_clipped).mul(0.5)
            value_loss = value_loss.mean()
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
                    if total_sampled > self.config.train_break_after_num_sampled * num_samples:
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

        all_states = []
        all_actions = []
        all_log_probs = []
        all_gaes = []
        all_values = []
        all_returns = []

        for train_env_player_id in self.config.player_ids:
            game_index, states, actions, log_probs, gaes, values, returns = self.train_env.dump(train_env_player_id, self.actor, self.critic,
                                                                                           self.summary_writer, f'train_{train_env_player_id}', self.global_step)


            self.logger.debug(f'dump: episode_buffers: '
                             f'completed_games: {len(game_index)}, '
                             f'experiences: {len(states)}, '
                             f'states: {states.shape}, '
                             f'actions: {actions.shape}, '
                             f'log_probs: {log_probs.shape}, '
                             f'values: {values.shape}, '
                             f'returns: {returns.shape}, '
                             f'gaes: {gaes.shape}')

            all_states.append(states)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_gaes.append(gaes)
            all_values.append(values)
            all_returns.append(returns)

        states = torch.cat(all_states, 0)
        actions = torch.cat(all_actions, 0)
        log_probs = torch.cat(all_log_probs, 0)
        gaes = torch.cat(all_gaes, 0)
        values = torch.cat(all_values, 0)
        returns = torch.cat(all_returns, 0)

        state_probs = np.ones(len(states), dtype=np.float32)

        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        if self.global_step % 10 == 0:
            unique_state_actions = defaultdict(list)

            for i, (state, action) in enumerate(zip(states, actions)):
                state_key = tuple(state.cpu().numpy().flatten().tolist())
                action_key = tuple(action.cpu().numpy().flatten().tolist())

                state_action_key = state_key + action_key

                unique_state_actions[state_action_key].append(i)

            self.summary_writer.add_scalars('train_iterations/samples', {
                'samples': len(states),
                'unique_state_actions': len(unique_state_actions),
            }, self.global_step)

            if False:
                new_states = []
                new_actions = []
                new_log_probs = []
                new_gaes = []
                new_values = []
                new_returns = []

                state_probs = np.ones(len(unique_state_actions), dtype=np.float32)
                for i, index in enumerate(unique_state_actions.values()):
                    state_probs[i] = len(index)
                    new_states.append(states[index[0]])
                    new_actions.append(actions[index[0]])

                    new_log_probs.append(log_probs[index].max(0)[0])
                    new_gaes.append(gaes[index].max(0)[0])
                    new_values.append(values[index].max(0)[0])
                    new_returns.append(returns[index].max(0)[0])

                states = torch.stack(new_states, 0).to(states.device)
                actions = torch.Tensor(new_actions).to(states.device)
                log_probs = torch.Tensor(new_log_probs).to(states.device)
                gaes = torch.Tensor(new_gaes).to(states.device)
                values = torch.Tensor(new_values).to(states.device)
                returns = torch.Tensor(new_returns).to(states.device)

                state_probs = np.power(state_probs, 0.6)

        state_probs /= state_probs.sum()
        self.set_training_mode(True)

        if self.config.enable_reference_move_training:
            self.optimize_actor_reference_moves()

        self.optimize_actor(states, actions, log_probs, gaes, state_probs)
        self.optimize_critic(states, returns, values, state_probs)

        self.global_step += 1

        return True

    def make_single_step_and_save(self, train_env: connectx_impl.ConnectX, player_id: int, actor: Actor, critic: Critic):
        actor.train(False)
        critic.train(False)

        game_index, game_states = train_env.current_states()
        if len(game_index) == 0:
            return

        with torch.no_grad():
            actions, log_probs, explorations = actor.dist_actions(player_id, game_states)

        new_states, rewards, dones = train_env.step(player_id, game_index, actions)

        episode_len = train_env.episode_len[game_index]
        truncated_indexes = torch.logical_and(episode_len + 1 == self.config.max_episode_len, dones != True)
        dones[truncated_indexes] = 1

        next_values = torch.zeros(len(game_index), dtype=torch.float32)
        if False:
            if truncated_indexes.sum() > 0:
                with torch.no_grad():
                    next_truncated_states = new_states[truncated_indexes, ...]
                    next_truncated_states = next_truncated_states.to(self.config.device)
                    nv = critic(next_truncated_states).detach().cpu()
                    next_values[truncated_indexes] = nv

        train_env.update_game_rewards(player_id, game_index, game_states, actions, log_probs, rewards, dones, next_values, explorations)

    def make_step(self):
        for player_id in self.config.player_ids:
            self.make_single_step_and_save(self.train_env, player_id, self.actor, self.critic)

    def fill_episode_buffer(self):
        self.train_env.reset()

        requested_states_size = self.config.batch_size * self.config.experience_buffer_to_batch_size_ratio
        winning_rate = float(self.max_eval_metric) / 100. * 1.5
        number_of_states = winning_rate * self.config.num_training_games * self.config.max_episode_len
        number_of_states_or_requested_states = max(number_of_states, requested_states_size)

        while True:
            self.make_step()

            total_games_completed = self.train_env.total_games_completed
            completed_games, completed_states = self.train_env.completed_games_and_states()
            running_games = len(self.train_env.running_index())

            self.logger.debug(f'fill_episode_buffer: '
                              f'total_games_completed: {total_games_completed}'
                              f'running_games: {running_games}, '
                              f'completed_games: {completed_games}, '
                              f'completed_states: {completed_states}, '
                              f'requested_size {requested_states_size}')

            if completed_states >= number_of_states_or_requested_states:
                break
            if running_games == 0:
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

            break


if __name__ == '__main__':
    main()
