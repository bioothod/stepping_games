import random

import numpy as np

from . import base_agent

class Agent(base_agent.Agent):
    def __init__(self, name, config):
        super().__init__(name, config)
        
    def check_winning_move(self, obs, column, player):
        grid = np.asarray(obs.board).reshape(self.config.rows, self.config.columns)
        next_grid = self.drop_piece(grid, column, player)
        has_won = self.check_winning_grid(next_grid, player)
        return has_won

    def action(self, obs):
        valid_moves = self.get_valid_moves(obs)
        current_player = obs.mark
        other_player = 2 if current_player == 1 else 1

        for column in valid_moves:
            is_winning = self.check_winning_move(obs, column, current_player)
            if is_winning:
                return column

        for column in valid_moves:
            is_winning = self.check_winning_move(obs, column, other_player)
            if is_winning:
                return column

        random_move = random.choice(valid_moves)
        return random_move


