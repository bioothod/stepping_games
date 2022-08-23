import multiprocessing as mp
import numpy as np

class Worker:
    def __init__(self, worker_id, make_env_fn, make_args_fn, seed):
        self.worker_id = worker_id
        self.make_env_fn = make_env_fn
        self.make_args_fn = make_args_fn
        self.seed = seed

        self.pipe = mp.Pipe()
        self.parent_end, self.worker_end = self.pipe
        
        self.worker = mp.Process(target=self.work, args=())
        self.worker.start()

        self.done = False

    def reset(self, **kwargs):
        self.send_msg(('', kwargs))

        ret = self.parent_end.recv()
        return ret

    def close(self, kwargs):
        self.send_msg(('close', kwargs))
        self.worker.join()
        
    def send_msg(self, msg):
        self.parent_end.send(msg)

    def work(self):
        env = self.make_env_fn(**self.make_args_fn, seed=self.seed)

        while True:
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
                break
        
class MultiprocessEnv:
    def __init__(self, make_env_fn, make_args_fn, seed, num_workers):
        self.workers = []
        self.worker_ids = list(range(num_workers))

        for worker_id in self.worker_ids:
            worker = Worker(worker_id, make_env_fn, make_args_fn, seed=seed+worker_id)
            self.workers.append(worker)

    def reset(self, workers, **kwargs):
        if workers is None:
            workers = self.worker_ids

        for worker_id in workers:
            self.workers[worker_id].reset(kwargs)

    def step(self, actions):
        if len(actions) != len(self.workers):
            raise ValueError(f'invalid actions: num_actions: {len(actions)}, workers: {len(self.workers)}: these must be equal')

        for worker, action in zip(self.workers, actions):
            worker.send_msg(('step', action))

        results = []
        for worker in self.workers:
            state, reward, done, info = worker.parent_end.recv()
            results.append((state, float(reward), float(done), info))

        results = np.array(results).T
        ret = [np.stack(block).squeeze() for block in results]
        return ret

    def close(self, **kwargs):
        for worker in self.workers:
            worker.close()
