import os
import random

from copy import deepcopy

import numpy as np
import torch

#import mcts

def create_submission_agent(checkpoint_path):
    from submission.model_ppo6 import Actor as eval_agent_model
    from submission.model_ppo6 import default_config

    agent = eval_agent_model(default_config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    agent.load_state_dict(checkpoint['actor_state_dict'])

    agent.train(False)

    return agent

class MultiprocessEval:
    def __init__(self, config, num_evaluations, agent_name, summary_writer, global_step):
        import gym_env
        from multiprocess_env import MultiprocessEnv

        self.summary_writer = summary_writer
        self.global_step = global_step

        self.agent_name = agent_name
        self.num_evaluations = num_evaluations

        self.device = config.device

        make_args_fn = lambda: {}
        def make_env_fn(seed=None):
            import random

            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            if config.train_player_id == 1:
                pair = [None, self.agent_name]
            else:
                pair = [self.agent_name, None]

            return gym_env.ConnectXGym(config, pair)

        eval_num_workers = min(num_evaluations, 50)
        self.eval_env = MultiprocessEnv('eval', make_env_fn, make_args_fn, config, config.eval_seed, eval_num_workers)

    def close(self):
        self.eval_env.close()

    def evaluate(self, train_agent):
        train_agent.set_training_mode(False)
        evaluation_rewards = []

        while len(evaluation_rewards) < self.num_evaluations:
            states = self.eval_env.reset()
            worker_ids = self.eval_env.worker_ids

            batch_size = len(states)
            episode_rewards = [0.0] * batch_size

            while True:
                states = torch.from_numpy(states).to(self.device)

                actions, _, _ = train_agent.actor.dist_actions(states)
                actions = actions.detach().cpu().numpy()
                new_states, rewards, dones, infos = self.eval_env.step(worker_ids, actions)

                ready_states = []
                ret_worker_ids = []
                for worker_id, new_state, reward, done in zip(worker_ids, new_states, rewards, dones):
                    if not done:
                        ready_states.append(new_state)
                        ret_worker_ids.append(worker_id)

                    episode_rewards[worker_id] += reward

                if len(ready_states) == 0:
                    break

                states = np.array(ready_states, dtype=np.float32)
                worker_ids = ret_worker_ids

            evaluation_rewards += episode_rewards

        last = evaluation_rewards[-self.num_evaluations:]
        wins = int(np.count_nonzero(np.array(last) >= 1) / len(last) * 100)

        return wins, evaluation_rewards

class DNNEval:
    def __init__(self, config, num_evaluations, agent, summary_writer, global_step):
        import connectx_impl

        self.summary_writer = summary_writer
        self.global_step = global_step

        self.num_evaluations = num_evaluations
        self.agent = agent

        self.train_player_id = config.train_player_id
        self.player_ids = config.player_ids
        self.device = config.device

        self.max_mean_episode_reward = -float('inf')

        config = deepcopy(config)
        config.num_training_games = num_evaluations
        self.eval_env = connectx_impl.ConnectX(config, None, self.summary_writer, 'eval', self.global_step)

    def close(self):
        self.eval_env.close()

    def evaluate(self, train_agent):
        train_agent.set_training_mode(False)

        self.eval_env.reset()

        while True:
            for player_id in self.player_ids:
                game_index, states = self.eval_env.current_states(player_id)
                if len(game_index) == 0:
                    break

                if player_id == self.train_player_id:
                    states = states.to(self.device)
                    actions, log_probs, explorations = train_agent.actor.dist_actions(states)
                else:
                    eval_states = self.eval_env.games[game_index]
                    if player_id == 2:
                        eval_states = self.eval_env.make_opposite(eval_states)
                    actions, log_probs, explorations = self.agent.dist_actions(eval_states)

                states, rewards, dones = self.eval_env.step(player_id, game_index, actions)

                self.eval_env.update_game_rewards(player_id, game_index, states, actions, log_probs, rewards, dones, torch.zeros_like(rewards), explorations)

            completed_index = self.eval_env.completed_index()
            if len(completed_index) >= self.num_evaluations:
                break

        completed_index = self.eval_env.completed_index()

        evaluation_rewards = []
        evaluation_explorations = []
        player_stat = self.eval_env.player_stats[self.train_player_id]
        for game_id in completed_index:
            episode_len = player_stat.episode_len[game_id]

            reward = player_stat.rewards[game_id, :episode_len].sum(0)
            exploration = player_stat.explorations[game_id, :episode_len].float().sum() / float(episode_len) * 100

            evaluation_rewards.append(reward)
            evaluation_explorations.append(exploration)

        wins = int(np.count_nonzero(np.array(evaluation_rewards) >= 1) / len(evaluation_rewards) * 100)
        mean_evaluation_reward = float(np.mean(evaluation_rewards))
        self.max_mean_episode_reward = max(self.max_mean_episode_reward, mean_evaluation_reward)

        self.summary_writer.add_scalar(f'eval/wins', wins, self.global_step)
        self.summary_writer.add_scalars(f'eval/episode_rewards', {
            'mean': mean_evaluation_reward,
            'max': self.max_mean_episode_reward,
        }, self.global_step)
        self.summary_writer.add_scalar(f'eval/episode_len', self.eval_env.episode_len.float().mean(), self.global_step)
        self.summary_writer.add_scalar(f'eval/exploration', np.mean(evaluation_explorations), self.global_step)

        return wins, evaluation_rewards

class Evaluate:
    def __init__(self, config, logger, num_evaluations, eval_agent, summary_writer, global_step):
        self.eval_seed = config.eval_seed

        self.logger = logger
        self.config = deepcopy(config)
        self.config['num_training_games'] = num_evaluations

        if isinstance(eval_agent, str):
            self.eval_obj = MultiprocessEval(config, num_evaluations, eval_agent, summary_writer, global_step)
        else:
            self.eval_obj = DNNEval(config, num_evaluations, eval_agent, summary_writer, global_step)

    def close(self):
        self.eval_obj.close()

    def evaluate(self, train_agent):
        torch.manual_seed(self.eval_seed)
        np.random.seed(self.eval_seed)
        random.seed(self.eval_seed)

        return self.eval_obj.evaluate(train_agent)

        mcts_agent = mcts.MCTSNaive(self.config.train_player_id, self.config, 10, train_agent)
        return self.eval_obj.evaluate(mcts_agent)
