RL algorithms for solving step-by-step games with scalable self-play
and against algorithmic agents.

* working on connectx deep-learning solver which could beat negamax,
  n-step lookahead (3 and 4 steps) and monte-carlo algorithms (all are
  implemented in the tree)
* GPU connectX implementation
* DDQN and PPO implementations suitable for the self-play games
* PPO with simple3 (conv+linear) model trained for several days of
  self-play beats negamax in 90-100% of the cases (depending on the seed)
