import math
import time
import random

from . import base_agent

class Game:
    def __init__(self, size=(6, 7), connect_size=4, position=None, turn=True, winner=None, terminal=False, moves=None):
        if position is None:
            self.position = (((0,) * size[0],) * size[1])
        else:
            self.position = position
        if moves is None:
            self.moves = []
        else:
            self.moves = moves
        self.turn, self.winner, self.terminal = turn, winner, terminal
        self.size, self.connect_size = size, connect_size

    @staticmethod
    def visualize_board(board):
        board_rows = [[] for _ in range(board.size[0])]
        for col in range(len(board.position)):
            for square in range(len(board.position[col])):
                board_rows[square].append(board.position[col][square])
        numbers_y, numbers_x = list(reversed(list(map(lambda x: x + 1, range(board.size[0]))))), list(reversed(list(map(lambda x: x + 1, range(board.size[1])))))
        max_length_y, max_length_x = len(str(numbers_y[0])), len(str(numbers_x[0]))
        # The top border (e.g. ┌———┬————┬————┬————┬————┐)
        result = "┌——" + ("—" * max_length_y)
        for _ in range(len(numbers_x)):
            result += "┬————"
        result += "┐\n"
        # Each rank, from top to bottom except the bottommost rank
        for index in range(len(numbers_y) - 1):
            result += f"│ {numbers_y[index]}{' ' * (max_length_y - len(str(numbers_y[index])))} │"  # The rank number
            # Add the square cells
            for square in board_rows[index]:
                result += VALUES[square] + "│"
            result += "\n"
            # Add the horizontal separator
            result += f"┝ {' ' * max_length_y} ┽"
            for _ in range(len(board_rows[index]) - 1):
                result += "————┼"
            result += "————┤\n"
        # Add the final rank with a special character
        result += f"│ 1{' ' * (max_length_y - 1)} │"
        for square in board_rows[-1]:
            result += VALUES[square] + "│"
        result += "\n"
        result += f"├——{'—' * max_length_y}┼"
        for _ in range(len(board_rows[index]) - 1):
            result += "————╁"
        result += "————┤\n"
        # Add the file numbers
        result += f"│  {' ' * max_length_y}│"
        for number in numbers_x[::-1]:
            result += f" {' ' * (max_length_x - len(str(number)))}{number} {' ' if number != numbers_x[0] else ''}{' ' if max_length_x == 1 else ''}"  # Append the file number with the correct spacing (on the last number, remove a space at the end for the border)
        result += f"│\n└—{'—' * max_length_y}—┴"
        for _ in range(len(numbers_x) - 1):
            result += "————┸"
        result += "————┘"
        return result

    @staticmethod
    def has_player_won(position):
        for color in [1, 2]:
            if position.get_n_in_a_row(position, color, position.connect_size):
                return color
        return None

    @staticmethod
    def is_tie(board):
        """Returns True if the game is tied, False otherwise."""
        for column in board.position:
            if column[0] == 0:
                return False
        return True

    @staticmethod
    def make_move(board, index):
        position = board.position
        for i in range(len(board.position[index - 1]))[::-1]:
            if not board.position[index - 1][i]:
                position = position[:index - 1] + (position[index - 1][:i] + (board.turn,) + position[index - 1][i + 1:],) + position[index:]
                turn = 3 - board.turn
                # winner/is_terminal variables: re-write functions here
                winner = Game.has_player_won(Game(position=position))
                terminal = (winner is not None) or Game.is_tie(Game(position=position))
                break
        return Game(position=position, turn=turn, winner=winner, terminal=terminal, moves=board.moves + [index])

    @staticmethod
    def get_n_in_a_row(board, player, n):
        """Returns the number of n-in-a-rows for the given player."""
        # TODO: (maybe) add a variable for occupied columns
        result = 0
        # Check vertical lines
        for column in board.position:
            for square in range(0, board.size[0] - n + 1):  # step parameter of range function should be (n - 1) (optimization)?
                for i in column[square:square + n]:
                    if i != player:
                        break
                else:
                    result += 1

        # Check horizontal lines
        for col_index in range(board.size[1] - n + 1):
            for square_index in range(board.size[0]):
                for i in board.position[col_index:col_index + n]:
                    if i[square_index] != player:
                        break
                else:
                    result += 1

        # Check diagonal lines
        for column_index in range(board.size[1] - n + 1):
            for row_index in range(board.size[0] - n + 1):
                for x in range(n):
                    if board.position[column_index + x][row_index + x] != player:
                        break
                else:
                    result += 1

            for row_index in range(n - 1, board.size[0]):
                for x in range(n):
                    if board.position[column_index + x][row_index - x] != player:
                        break
                else:
                    result += 1
        return result

    @staticmethod
    def legal_moves(board):
        moves = []
        for column_index in range(len(board.position)):
            if not board.position[column_index][0]:
                moves.append(column_index + 1)
        return moves


class MonteCarlo:
    def __init__(self):
        self.outcomes = {}
        self.visits = {}
        self.tree = {}

    @staticmethod
    def make_legal_moves(board):
        if board.terminal:
            return set()
        return {board.make_move(board, i) for i in board.legal_moves(board)}

    @staticmethod
    def make_random_move(board):
        if board.terminal:
            return None
        return board.make_move(board, random.choice(board.legal_moves(board)))

    @staticmethod
    def evaluate(board):
        if bool(board.turn - 1) == board.winner:
            return 0
        return 0.5

    def choose(self, node):
        if node not in self.tree:
            return MonteCarlo.make_random_move(node)

        def score(n):
            if self.visits[n] == 0:
                return float("-inf")
            return self.outcomes[n] / self.visits[n]

        return max(self.tree[node], key=score)

    def rollout(self, node):
        path = self.select(node)
        leaf = path[-1]
        self.expand(leaf)
        outcome = self.simulate(leaf)
        self.backpropagate(path, outcome)

    def select(self, node, path=None):
        if path is None:
            path = []
        path.append(node)
        if node not in self.tree or not self.tree[node]:
            return path
        unexplored = self.tree[node] - self.tree.keys()
        if unexplored:
            path.append(unexplored.pop())
            return path
        return self.select(self.uct(node), path)

    def expand(self, node):
        if node not in self.tree:
            self.tree[node] = MonteCarlo.make_legal_moves(node)

    @staticmethod
    def simulate(node, invert=True):
        if node.terminal:
            outcome = MonteCarlo.evaluate(node)
            if invert:
                return 1 - outcome
            return outcome
        return MonteCarlo.simulate(MonteCarlo.make_random_move(node), not invert)

    def backpropagate(self, path, outcome):
        for node in path[::-1]:
            if node in self.visits:
                self.visits[node] += 1
            else:
                self.visits[node] = 1
            if node in self.outcomes:
                self.outcomes[node] += outcome
            else:
                self.outcomes[node] = outcome
            outcome = 1 - outcome

    def uct(self, node):
        log = math.log(self.visits[node])
        return max(self.tree[node], key=lambda n: (self.outcomes[n] / self.visits[n]) + (2 * math.sqrt(log / self.visits[n])))

class Agent(base_agent.Agent):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self, observation):
        # Use a simple opening book for the first move
        if observation.board == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            return 3, -1
        elif observation.board == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]:
            return 1, -1
        elif observation.board == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]:
            return 2, -1
        elif observation.board == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]:
            return 3, -1
        elif observation.board == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]:
            return 3, -1
        elif observation.board == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]:
            return 4, -1
        elif observation.board == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]:
            return 4, -1
        elif observation.board == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
            return 5, -1
        END_TIME = time.time() + 7.25
        VALUES = {0: "    ", 1: "\033[31m ◉  \033[0m", 2: "\033[93m ◉  \033[0m"}


        position = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        index = 0
        position1 = [[]]
        for i in observation["board"]:
            if index >= self.config["columns"]:
                index = 0
                position1.append([])
            position1[-1].append(i)
            index += 1
        index = 0
        for i in range(len(position1)):
            for x in range(len(position1[i])):
                position[x][i] = position1[i][x]
        for i in range(len(position)):
            position[i] = tuple(position[i])
        position = tuple(position)
        game = Game(position=position, size=(self.config["rows"], self.config["columns"]), connect_size=self.config["inarow"], turn=observation["mark"])
        mcts = MonteCarlo()
        rollouts = "inf (while time is left)"
        if observation["remainingOverageTime"] > 22:
            while True:
                if time.time() >= END_TIME:
                    break
                mcts.rollout(game)
        elif observation["remainingOverageTime"] > 15:
            for i in range(2500):
                mcts.rollout(game)
            rollouts = 2500
        elif observation["remainingOverageTime"] > 10:
            for i in range(2000):
                mcts.rollout(game)
            rollouts = 2000
        elif observation["remainingOverageTime"] < 5:
            import random
            return [random.choice(list(range(self.config["columns"]))), rollouts, True]
        else:
            rollouts = 1500
            for i in range(1500):
                mcts.rollout(game)
        return [mcts.choose(game).moves[-1] - 1, rollouts, False]


    def action(self, observation):
        #print("INFO position " + str(observation["board"]))
        #if "remainingOverageTime" in observation:
        #    print("INFO remaining_time " + str(observation["remainingOverageTime"]))
        observation["remainingOverageTime"] = 15
        agent_string = self.run(observation)
        #print("INFO rollouts " + str(agent_string[1]) + "\nINFO move " + str(agent_string[0]))
        if agent_string[1] == -1:
            #print("INFO opening")
            return agent_string[0]
        if agent_string[2]:
            #print("WARN remaining_time < 5s, moves will be random")
            pass
        return agent_string[0]
