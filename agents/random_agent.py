import random

from . import base_agent

class Agent(base_agent.Agent):
    def __init__(self, name, config):
        super().__init__(name, config)

    def action(self, state):
        valid_moves = self.get_valid_moves(state)

        chosen_move = random.choice(valid_moves)
        return chosen_move

