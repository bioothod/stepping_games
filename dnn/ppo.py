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

class PPO(train_selfplay.BaseTrainer):
    def __init__(self):
        self.config = edict({
            'player_ids': [1, 2],
            
            'load_checkpoints_dir': 'checkpoints_simple3_ppo_7',
            'checkpoints_dir': 'checkpoints_simple3_ppo_9',
            'eval_checkpoint_path': 'checkpoints_simple3_ppo_7/ppo_100.ckpt',

            'eval_after_train_steps': 20,

            'max_episode_len': 42,

            'policy_optimization_steps': 10,
            'policy_clip_range': 0.1,
            'policy_stopping_kl': 0.6,

            'value_optimization_steps': 10,
            'value_clip_range': float('inf'),
            'value_stopping_mse': 0.7,

            'entropy_loss_weight': 0.2,

            'gamma': 0.99,
            'tau': 0.97,

            'init_lr': 1e-5,

            'num_games_to_stop_training_state_model': 10_000_000,

            'num_features': 512,
            'hidden_dims': [128],

            'max_gradient_norm': 1.0,

            'num_training_games': 1024*2,

            'batch_size': 1024*8,
            'experience_buffer_to_batch_size_ratio': 2,
        })

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

        self.logger.info(f'actor:\n{print_networks("actor", self.actor, verbose=True)}')
        self.logger.info(f'critic:\n{print_networks("critic", self.critic, verbose=True)}')

        model_loaded = self.try_load(self.name, self)

        self.train_env = connectx_impl.ConnectX(self.config, self.critic)

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
            batch_indexes = np.random.choice(num_samples, min(num_samples, self.config.batch_size), replace=False)

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
                start_index = 0
                pred_log_probs_all = []
                while start_index < len(states):
                    batch_states = states[start_index : start_index + self.config.batch_size, ...]
                    batch_actions = actions[start_index : start_index + self.config.batch_size]
                    pred_log_probs, _ = self.actor.get_predictions(batch_states, batch_actions)
                    pred_log_probs_all.append(pred_log_probs)
                    start_index += len(batch_states)

                pred_log_probs_all = torch.cat(pred_log_probs_all)
                kl = (log_probs - pred_log_probs_all).mean()
                kls.append(kl.item())
                if kl.item() > self.config.policy_stopping_kl:
                    break

        mean_policy_loss = np.mean(policy_losses)
        mean_entropy_loss = np.mean(entropy_losses)
        mean_total_loss = np.mean(total_losses)
        mean_kl = np.mean(kls)
        self.logger.debug(f'optimize_actor: '
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
        for _ in range(self.config.value_optimization_steps):
            batch_indexes = np.random.choice(num_samples, min(num_samples, self.config.batch_size), replace=False)

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

            value_losses.append(value_loss.item())
            with torch.no_grad():
                pred_values_all = []
                start_index = 0
                while start_index < len(states):
                    batch_states = states[start_index : start_index + self.config.batch_size, ...]
                    pred_values = self.critic(batch_states)
                    pred_values_all.append(pred_values)
                    start_index += len(batch_states)

                pred_values_all = torch.cat(pred_values_all)
                mse_val = (values - pred_values_all).pow(2).mul(0.5).mean()
                mse_ret = (returns - pred_values_all).pow(2).mul(0.5).mean()

                mse_values.append(mse_val.item())
                mse_returns.append(mse_ret.item())

                if mse_val.item() > self.config.value_stopping_mse or mse_ret.item() > self.config.value_stopping_mse:
                    break

        mean_value_loss = np.mean(value_losses)
        mean_mse_values = np.mean(mse_values)
        mean_mse_returns = np.mean(mse_returns)
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

        return True

    def make_single_step(self, player_id, game_index, states):
        states = states.to(self.config.device)

        with torch.no_grad():
            actions, log_probs, explorations = self.actor.dist_actions(states)

        new_states, rewards, dones = self.train_env.step(player_id, game_index, actions)
        new_states = self.train_env.make_state(player_id, new_states)

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
        game_index, states = self.train_env.current_states()
        if len(states) == 0:
            #self.logger.info(f'player_id: {player_id}: states: {len(states)}')
            return

        states = self.train_env.make_state(player_id, states)

        episode_len = self.train_env.episode_len[game_index]

        step = self.make_single_step(player_id, game_index, states)

        truncated_indexes = torch.logical_and(episode_len + 1 == self.config.max_episode_len, step.dones != True)
        step.dones[truncated_indexes] = 1

        next_values = torch.zeros(len(step.states), dtype=torch.float32)
        if len(truncated_indexes) > 0:
            with torch.no_grad():
                next_truncated_states = step.new_states[truncated_indexes, ...]
                next_truncated_states = next_truncated_states.to(self.config.device)
                nv = self.critic(next_truncated_states).detach().cpu()
                next_values[truncated_indexes] = nv

        self.train_env.update_game_rewards(player_id, game_index, step.states, step.actions, step.log_probs, step.rewards, step.dones, step.explorations, next_values)

    def make_step(self):
        self.make_single_step_and_save(1)
        self.make_single_step_and_save(2)


        if self.train_env.total_games_completed > self.config.num_games_to_stop_training_state_model:
            if self.actor.train_state_features:
                self.logger.info(f'completed_games: {len(self.train_env.completed_games)}, '
                                 f'num_games_to_stop_training_state_model: {self.config.num_games_to_stop_training_state_model}: '
                                 f'finishing training the state model')

            self.actor.train_state_features = False

    def fill_episode_buffer(self):
        self.train_env.reset()

        while len(self.train_env.running_index()) > 0:
            self.make_step()

            completed_games, completed_states = self.train_env.completed_games_and_states()

            requested_states_size = self.config.batch_size * self.config.experience_buffer_to_batch_size_ratio
            self.logger.debug(f'fill_episode_buffer: '
                             f'running_games: {len(self.train_env.running_index())}, '
                             f'completed_games: {completed_games}, '
                             f'completed_states: {completed_states}, '
                             f'requested_size {requested_states_size}')

            if False and completed_states >= requested_states_size:
                break


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
