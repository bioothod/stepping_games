import multiprocessing as mp
import numpy as np

from logger import setup_logger

class Worker:
    def __init__(self, name, worker_id, make_env_fn, make_args_fn, seed, logfile):
        self.name = f'{name}_{worker_id}'
        self.worker_id = worker_id
        self.make_env_fn = make_env_fn
        self.make_args_fn = make_args_fn
        self.seed = seed
        
        self.logger = setup_logger(self.name, logfile, False)

        self.pipe = mp.Pipe()
        self.parent_end, self.worker_end = self.pipe
        
        self.worker = mp.Process(target=self.work, args=())
        self.worker.start()

        self.done = False
        
        self.logger.info(f'{self.name}: started worked {worker_id}')

    def reset(self, **kwargs):
        self.send_msg(('reset', kwargs))

        ret = self.parent_end.recv()
        return ret

    def close(self, **kwargs):
        self.send_msg(('close', kwargs))
        self.worker.terminate()
        self.worker.join()
        
    def send_msg(self, msg):
        self.parent_end.send(msg)

    def do_step(self, env):
        cmd, kwargs = self.worker_end.recv()

        if cmd == 'reset':
            ret = env.reset(**kwargs)
            self.worker_end.send(ret)
        elif cmd == 'step':
            ret = env.step(**kwargs)
            self.worker_end.send(ret)
        elif cmd == '_past_limit':
            ret = env._elapsed_steps >= env._max_episode_steps
            self.worker_end.send(ret)
        else:
            env.close(**kwargs)
            del env
            self.worker_end.close()
            ret = None

        return ret

    def work(self):
        env = self.make_env_fn(**self.make_args_fn(), seed=self.seed)

        while True:
            try:
                ret = self.do_step(env)
                if ret is None:
                    break
            except KeyboardInterrupt:
                env.close()
                self.worker_end.close()
                break
        
class MultiprocessEnv:
    def __init__(self, name, make_env_fn, make_args_fn, config, seed, num_workers):
        self.name = name
        
        self.workers = []
        self.worker_ids = list(range(num_workers))
        
        self.logger = setup_logger(f'multiprocess_env_{name}', config.logfile, config.log_to_stdout)

        for worker_id in self.worker_ids:
            worker_seed = seed + worker_id
            worker = Worker(self.name, worker_id, make_env_fn, make_args_fn, seed=worker_seed, logfile=config.logfile)
            self.workers.append(worker)

        self.logger.info(f'{self.name}: started multiprocessing environment, num_workers: {num_workers}')

    def reset(self, workers=None, **kwargs):
        if workers is None:
            workers = self.worker_ids

        ret_array = []
        for worker_id in workers:
            ret = self.workers[worker_id].reset(**kwargs)
            ret_array.append(ret)

        ret_array = np.array(ret_array)
        return ret_array

    def step(self, worker_ids, actions):
        if len(actions) != len(worker_ids):
            raise ValueError(f'{self.name}: invalid actions: num_actions: {len(actions)}, workers: {len(worker_ids)}: these must be equal')

        for worker_id, action in zip(worker_ids, actions):
            worker = self.workers[worker_id]
            worker.send_msg(('step', {'action': action}))

        ret_states = []
        ret_rewards = []
        ret_dones = []
        ret_infos = []
        
        for worker_id in worker_ids:
            worker = self.workers[worker_id]
            state, reward, done, info = worker.parent_end.recv()
            
            ret_states.append(state)
            ret_rewards.append(reward)
            ret_dones.append(done)
            ret_infos.append(info)

        ret_states = np.array(ret_states)
        ret_rewards = np.array(ret_rewards)
        ret_dones = np.array(ret_dones)
        
        return ret_states, ret_rewards, ret_dones, ret_infos

    def close(self, **kwargs):
        for worker in self.workers:
            worker.close(**kwargs)
