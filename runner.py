import random

from easydict import EasyDict as edict

import numpy as np
#import torch

import kaggle_environments as kaggle

import agents
from logger import setup_logger

class ConnectX:
    def __init__(self):
        self.logger = setup_logger('kaggle', logfile=None, log_to_stdout=True)
        
        self.config = edict({
            'rows': 6,
            'columns': 7,
            'inarow': 4
        })
        
        self.env = kaggle.make("connectx", debug=True, configuration={ "actTimeout": 9999999999 })
    
    def get_win_percentages(self, agent1_obj, agent2_obj, n_rounds):
        self.logger.info(f'agent1: {agent1_obj}, agent2: {agent2_obj}, rounds: {n_rounds}')
        def convert_agent(agent_obj):
            if type(agent_obj) == str:
                return agent_obj

            return lambda obs, config: agent_obj.action(obs)

        agent1 = convert_agent(agent1_obj)
        agent2 = convert_agent(agent2_obj)
        
        outcomes = kaggle.evaluate("connectx", [agent1, agent2], self.config, [], n_rounds//2)
        outcomes += [[b,a] for [a,b] in kaggle.evaluate("connectx", [agent2, agent1], self.config, [], n_rounds-n_rounds//2)]

        agent1_winning = np.round(outcomes.count([1,-1])/len(outcomes), 2)
        agent2_winning = np.round(outcomes.count([-1,1])/len(outcomes), 2)
        agent1_invalid = outcomes.count([None, 0])
        agent2_invalid = outcomes.count([0, None])

        self.logger.info(f'agent1: {agent1_obj}: winning: {agent1_winning}, invalid: {agent1_invalid}')
        self.logger.info(f'agent2: {agent2_obj}: winning: {agent2_winning}, invalid: {agent2_invalid}')

def main():
    connect = ConnectX()

    agent2 = agents.monte_carlo_agent.Agent('monte_carlo', connect.config)
    #agent2 = agents.minimax_agent.Agent('minimax', connect.config)
    #agent1 = 'negamax'
    #agent1 = agents.monte_carlo_agent.Agent('monte_carlo', connect.config)
    agent1 = agents.lookahead_agent.Agent('lookahead', connect.config)
    connect.get_win_percentages(agent1, agent2, 100)

if __name__ == '__main__':
    main()
