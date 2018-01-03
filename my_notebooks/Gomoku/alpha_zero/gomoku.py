import numpy as np
from matplotlib import pyplot as plt
import math

class Gomoku:

    def __init__(self, shape):
        self.shape = shape
        self.reset()

    def reset(self):
        self.last_player = 1
        self.board = np.stack((np.zeros(self.shape), np.zeros(self.shape),np.ones(self.shape)), axis=2)
        self.action_stack = []
        self.group_map = np.zeros((2, self.shape[0], self.shape[1], 4))
        self.number_of_steps = 0

    def draw(self):
        plt.imshow(self.board)

    def char_draw(self):
        board_string = '*'*(2*self.board.shape[0]+2) + "\n"
        for i in range(self.board.shape[1]):
            board_string += '*'
            for j in range(self.board.shape[0]):
                if self.board[i,j,0] == 1:
                    board_string += 'X '
                elif self.board[i,j,1] == 1:
                    board_string += 'O '
                else:
                    board_string += '. '
            board_string += "*\n"
        board_string += '*'*(2*self.board.shape[0]+2) + "\n"

        print(board_string)



    def get_occupied(self):
        return np.sum(self.board[:,:,0:2], axis=2)

    def take_action(self, action):
        if np.sum(self.board[action][0:2]) > 0:
            raise Exception('Invalid action' + str(action))

        self.last_player = 1 - self.last_player
        self.board[action][self.last_player] = 1
        self.number_of_steps += 1

        self.__update_group_map(self.group_map[self.last_player], action)

        self.action_stack.append(action)

    def revert(self):
        action = self.action_stack.pop()
        self.board[action][self.last_player] = 0
        self.__update_group_map(self.group_map[self.last_player], action, -1)
        self.number_of_steps -= 1

        self.last_player = 1 - self.last_player


    def __update_group_map(self, gmap, action, direction = 1):
        deltas = [
            [-2,-2, 0],[-1,-1, 0],[ 0, 0, 0],[ 1, 1, 0],[ 2, 2, 0],
            [-2, 0, 1],[-1, 0, 1],[ 0, 0, 1],[ 1, 0, 1],[ 2, 0, 1],
            [-2, 2, 2],[-1, 1, 2],[ 0, 0, 2],[ 1,-1, 2],[ 2,-2, 2],
            [ 0,-2, 3],[ 0,-1, 3],[ 0, 0, 3],[ 0, 1, 3],[ 0, 2, 3],
        ]

        for delta in deltas:
            ax = action[0] + delta[0]
            ay = action[1] + delta[1]

            if ax >= 0 and ax < self.shape[0] and ay >= 0 and ay < self.shape[1]:
                gmap[ax, ay, delta[2]] += direction

    def get_last_action(self):
        return self.action_stack[-1]

    def get_current_player_id(self):
        return 1 - self.last_player

    def get_last_player_id(self):
        return self.last_player

    def get_state_for_current_player(self):
        return self.get_state_for_player(1 - self.last_player)

    def get_state_for_player(self, player):
        return self.convert_state_for_player(self.board, player)

    def get_state(self):
        return self.board

    def get_shape(self):
        return self.shape

    def get_group_map(self):
        return self.group_map

    def get_number_of_steps(self):
        return self.number_of_steps

    def get_middle(self):
        return (math.floor(self.shape[0]/2), math.floor(self.shape[0]/2))

    def convert_state_for_player(self, board, player):
        result = np.copy(board)

        if player == 1:
            tmp = np.copy(result[:,:,0])
            result[:,:,0] = result[:,:,1]
            result[:,:,1] = tmp

        return result

    def game_over(self):
        return self.won(0) or self.won(1) or np.sum(1-self.get_occupied()) == 0

    def winner(self):
        if self.won(0):
            return 1

        if self.won(1):
            return -1

        return 0

    def winner_from_last_players_perspective(self):
        if self.won(self.last_player):
            return 1

        if self.won(1 - self.last_player):
            return -1

        return 0


    def won(self, player):
        return np.count_nonzero(self.group_map[player] == 5)
