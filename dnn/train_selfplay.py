import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True

import evaluate
import logger

class BaseTrainer:
    def __init__(self, config):
        config.train_seed = 444
        config.eval_seed = 555

        self.summary_writer = SummaryWriter(log_dir=config.tensorboard_log_dir)

        torch.manual_seed(config.train_seed)
        np.random.seed(config.train_seed)
        random.seed(config.train_seed)

        config.device = torch.device('cuda:0')
        config.rows = 6
        config.columns = 7
        config.inarow = 4
        config.player_ids = [1, 2]
        config.num_actions = config.columns
        config.observation_shape = [1, self.config.rows, self.config.columns]

        self.config = config

        os.makedirs(self.config.checkpoints_dir, exist_ok=True)

        self.config.logfile = os.path.join(self.config.checkpoints_dir, 'train.log')
        self.config.log_to_stdout = True
        self.logger = logger.setup_logger('t', self.config.logfile, log_to_stdout=self.config.log_to_stdout)

        config_message = []
        for k, v in self.config.items():
            config_message.append(f'{k:>32s}:{str(v):>16s}')
        config_message = '\n'.join(config_message)

        self.logger.info(f'config:\n{config_message}')

        self.num_evaluations_per_epoch = 100
        self.max_score_metric = 0.0
        self.max_eval_metric = 0.0
        self.max_mean_eval_metric = -float('inf')

        eval_agent_template = config.get('eval_agent_template')
        if eval_agent_template:
            self.eval_agent_name = eval_agent_template

            eval_agent = evaluate.create_submission_agent(eval_agent_template)
            self.evaluation = evaluate.Evaluate(config, self.logger, self.num_evaluations_per_epoch, eval_agent, self.summary_writer, self.global_step, 'eval')
        else:
            self.eval_agent_name = 'negamax'
            self.evaluation = evaluate.Evaluate(config, self.logger, self.num_evaluations_per_epoch, self.eval_agent_name, self.summary_writer, self.global_step, 'eval')

    def try_train(self):
        raise NotImplementedError('method @try_train() needs to be implemented')

    def run_epoch(self, train_env, train_agent):
        training_started = False

        for _ in range(self.config.eval_after_train_steps):
            train_agent.set_training_mode(True)

            training_started = self.try_train()

        if training_started:
            mean_rewards, std_rewards, mean_explorations, std_explorations, mean_timesteps, std_timesteps = train_env.completed_games_stats()
            if len(mean_rewards) == 0:
                return

            total_games_completed = train_env.total_games_completed

            train_agent.set_training_mode(False)
            eval_metric, eval_rewards = self.evaluation.evaluate(train_agent)
            mean_eval_score = np.mean(eval_rewards)

            best_score, good_score = self.evaluation.score_eval_ds.evaluate(train_agent.actor, debug=False)
            self.summary_writer.add_scalars('eval/score_metric', {
                'best_score': best_score,
                'good_score': good_score,
            }, self.global_step)

            if eval_metric >= self.max_eval_metric or best_score >= self.max_score_metric or mean_eval_score > self.max_mean_eval_metric:
                self.logger.info(f'games: {total_games_completed:6d}: '
                                 f'eval: '
                                 f'r: {mean_eval_score:7.4f} / {self.max_mean_eval_metric:7.4f}, '
                                 f'metric: {eval_metric:2.0f} / {self.max_eval_metric:2.0f}, '
                                 f'best_score: {best_score:.2f} / {self.max_score_metric:.2f}, good_score: {good_score:.2f}'
                                 )

            if eval_metric > 0 and eval_metric >= self.max_eval_metric:
                if eval_metric < 100:
                    self.max_eval_metric = eval_metric

                checkpoint_path = os.path.join(self.config.checkpoints_dir, f'{train_agent.name}_{eval_metric:.0f}.ckpt')
                train_agent.save(checkpoint_path)

                self.logger.info(f'eval_metric: {eval_metric:.0f}, saved {train_agent.name} -> {checkpoint_path}')

            if best_score >= self.max_score_metric:
                self.max_score_metric = best_score
                checkpoint_path = os.path.join(self.config.checkpoints_dir, f'{train_agent.name}_best_score.ckpt')
                train_agent.save(checkpoint_path)

                self.logger.info(f'max_score_metric: {best_score:.2f}, saved {train_agent.name} -> {checkpoint_path}')

            if mean_eval_score > self.max_mean_eval_metric:
                self.max_mean_eval_metric = mean_eval_score

                checkpoint_path = os.path.join(self.config.checkpoints_dir, f'{train_agent.name}_mean_score_improvement.ckpt')
                train_agent.save(checkpoint_path)
                self.logger.info(f'mean_eval_score: {mean_eval_score:.4f}, saved {train_agent.name} -> {checkpoint_path}')

    def try_load(self, name, model):
        max_metric = None
        checkpoint_path = None

        checkpoints_dir = self.config.get('load_checkpoints_dir', self.config['checkpoints_dir'])
        for checkpoint_fn in os.listdir(checkpoints_dir):
            if checkpoint_fn == f'{name}.ckpt':
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_fn)
                max_metric = 0.0
                continue

            metric, ext = os.path.splitext(checkpoint_fn)
            if ext != '.ckpt':
                continue

            if not metric.startswith(name):
                continue

            metric = metric.split('_')
            try:
                metric = float(metric[-1])
            except:
                continue

            if max_metric is None or metric > max_metric:
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_fn)
                max_metric = metric

        if checkpoint_path is not None and max_metric is not None:
            self.logger.info(f'{name}: loading checkpoint {checkpoint_path}, metric: {max_metric}')
            model.load(checkpoint_path)
            self.max_eval_metric = max_metric
            return True

        return False
        
    def stop(self):
        self.evaluation.close()
