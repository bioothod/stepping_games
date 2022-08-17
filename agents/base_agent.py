import numpy as np

class Agent:
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def __str__(self):
        return self.name

    def __call__(self, *args, **kvargs):
        return self.action(*args, **kvargs)
    
    def action(self, state):
        raise NotImplementedError(f'method @action is not implemented')
        
    def get_valid_moves(self, obs):
        grid = np.asarray(obs.board).reshape(self.config.rows, self.config.columns)
        first_row = grid[0]

        valid_moves = []
        for column_index in range(self.config.columns):
            if first_row[column_index] == 0:
                valid_moves.append(column_index)

        return valid_moves

    # Gets a board at the next step if an agent drops a piece in the selected column
    def drop_piece(self, grid, column, player):
        next_grid = grid.copy()
        for row in range(self.config.rows-1, -1, -1):
            if next_grid[row][column] == 0:
                break
        next_grid[row][column] = player
        return next_grid

    # Check if a particular `player` has won the game on a particular `grid`
    def check_winning_grid(self, grid, player):
        # horizontal
        for row in range(self.config.rows):
            for col in range(self.config.columns-(self.config.inarow-1)):
                window = list(grid[row, col:col+self.config.inarow])
                if window.count(player) == self.config.inarow:
                    return True

        # vertical
        for row in range(self.config.rows-(self.config.inarow-1)):
            for col in range(self.config.columns):
                window = list(grid[row:row+self.config.inarow, col])
                if window.count(player) == self.config.inarow:
                    return True

        # positive diagonal
        for row in range(self.config.rows-(self.config.inarow-1)):
            for col in range(self.config.columns-(self.config.inarow-1)):
                window = list(grid[range(row, row+self.config.inarow), range(col, col+self.config.inarow)])
                if window.count(player) == self.config.inarow:
                    return True

        # negative diagonal
        for row in range(self.config.inarow-1, self.config.rows):
            for col in range(self.config.columns-(self.config.inarow-1)):
                window = list(grid[range(row, row-self.config.inarow, -1), range(col, col+self.config.inarow)])
                if window.count(player) == self.config.inarow:
                    return True

        return False

