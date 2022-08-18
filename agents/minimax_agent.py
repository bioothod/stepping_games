import random

import numpy as np



from . import lookahead_agent

class Agent(lookahead_agent.Agent):
    def __init__(self, name, config):
        super().__init__(name, config)

    def score_move_minimax(self, grid, column, mark, nsteps):
        next_grid = self.drop_piece(grid, column, mark)
        score = self.minimax(next_grid, nsteps-1, False, mark, -np.Inf)
        return score

    def is_terminal_node(self, grid):
        if list(grid[0, :]).count(0) == 0:
            return True

        if self.check_winning_grid(grid, 1):
            return True

        if self.check_winning_grid(grid, 2):
            return True

        return False

    def minimax(self, node, depth, maximizing_player, player, up_value):
        valid_moves = [c for c in range(self.config.columns) if node[0][c] == 0]

        if depth == 0 or self.is_terminal_node(node):
            return self.get_heuristic(node, player)

        other_player = 1 if player == 2 else 2

        if maximizing_player:
            value = -np.Inf
            for column in valid_moves:
                child = self.drop_piece(node, column, player)
                value = max(value, self.minimax(child, depth-1, False, player, value))
                if value >= up_value:
                    break
        else:
            value = np.Inf
            for column in valid_moves:
                child = self.drop_piece(node, column, other_player)
                value = min(value, self.minimax(child, depth-1, True, player, value))
                if value <= up_value:
                    break

        return value

    def action(self, obs):
        if sum(obs.board) == 0:
            return 3

        valid_moves = [c for c in range(self.config.columns) if obs.board[c] == 0]

        grid = np.asarray(obs.board).reshape(self.config.rows, self.config.columns)

        n_steps = 4
        scores = dict(zip(valid_moves, [self.score_move_minimax(grid, column, obs.mark, n_steps) for column in valid_moves]))

        max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]

        return random.choice(max_cols)

