import random

import numpy as np

from . import base_agent

class Agent(base_agent.Agent):
    def __init__(self, name, config):
        super().__init__(name, config)

    def score_move(self, grid, column, player):
        next_grid = self.drop_piece(grid, column, player)
        score = self.get_heuristic(next_grid, player)
        return score

    def get_heuristic(self, grid, player):
        other_player = 1 if player == 2 else 2

        num2 = self.count_windows(grid, 2, player)
        num3 = self.count_windows(grid, 3, player)
        num4 = self.count_windows(grid, 4, player)
        num2_other = self.count_windows(grid, 2, other_player)
        num3_other = self.count_windows(grid, 3, other_player)

        score = 1e10 * num4 + 1e4 * num3 + 1e2 * num2 + -1 * num2_other + -1e6 * num3_other
        return score

    def check_window(self, window, num_discs, piece):
        return (window.count(piece) == num_discs and window.count(0) == self.config.inarow - num_discs)

    def count_windows(self, grid, num_discs, piece):
        num_windows = 0

        # horizontal
        for row in range(self.config.rows):
            for col in range(self.config.columns-(self.config.inarow-1)):
                window = list(grid[row, col:col+self.config.inarow])
                if self.check_window(window, num_discs, piece):
                    num_windows += 1

        # vertical
        for row in range(self.config.rows-(self.config.inarow-1)):
            for col in range(self.config.columns):
                window = list(grid[row:row+self.config.inarow, col])
                if self.check_window(window, num_discs, piece):
                    num_windows += 1

        # positive diagonal
        for row in range(self.config.rows-(self.config.inarow-1)):
            for col in range(self.config.columns-(self.config.inarow-1)):
                window = list(grid[range(row, row+self.config.inarow), range(col, col+self.config.inarow)])
                if self.check_window(window, num_discs, piece):
                    num_windows += 1

        # negative diagonal
        for row in range(self.config.inarow-1, self.config.rows):
            for col in range(self.config.columns-(self.config.inarow-1)):
                window = list(grid[range(row, row-self.config.inarow, -1), range(col, col+self.config.inarow)])
                if self.check_window(window, num_discs, piece):
                    num_windows += 1

        return num_windows

    def action(self, obs):
        valid_moves = self.get_valid_moves(obs)

        grid = np.asarray(obs.board).reshape(self.config.rows, self.config.columns)

        player = obs.mark
        min_score = -1e20
        best_move = None
        for column in valid_moves:
            score = self.score_move(grid, column, player)
            if score > min_score:
                min_score = score
                best_move = column

        return best_move

