import numpy as np
from matplotlib import pyplot as plt
import math

class Gomoku:

    def __init__(self, shape):
        self.shape = shape
        self.reset()

    def reset(self):
        self.last_player = 1
        base = np.ones(self.shape)
        base[math.floor(self.shape[0]/2), math.floor(self.shape[1]/2)] += 1
        self.board = np.stack((np.zeros(self.shape), np.zeros(self.shape),base), axis=2)
        self.action_stack = []
        self.number_of_steps = 0
        self.winner = 0

    def draw(self):
        plt.imshow(self.board)

    def char_draw_as_string(self):
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
        return board_string

    def char_draw(self):
        print(self.char_draw_as_string())



    def get_occupied(self):
        return np.sum(self.board[:,:,0:2], axis=2)

    def take_action(self, action):
        if self.winner != 0:
            raise Exception('Action initiatied after game was already over')

        if np.sum(self.board[action][0:2]) > 0:
            raise Exception('Invalid action' + str(action))

        self.last_player = 1 - self.last_player
        self.board[action][self.last_player] = 1
        self.number_of_steps += 1


        self.__update_winner(action)

        self.action_stack.append(action)

    def revert(self):
        action = self.action_stack.pop()
        self.board[action][self.last_player] = 0
        self.number_of_steps -= 1
        self.winner = 0

        self.last_player = 1 - self.last_player


    def __update_winner(self, action):

        for direction in [(0,1),(1,1),(1,0),(1,-1)]:
            c = self.__same_in_direction(action, direction)
            if c >= 5:
                self.winner = 1 if self.last_player == 0 else -1
                return

    def __same_in_direction(self, action, direction):

        last_players_pieces = self.board[:,:,self.last_player]
        c = 0

        cursor = action
        while self.__valid_index(cursor) and last_players_pieces[cursor] == 1:
            cursor = (cursor[0] + direction[0], cursor[1] + direction[1])
            c += 1

        cursor = action
        while self.__valid_index(cursor) and last_players_pieces[cursor] == 1:
            cursor = (cursor[0] - direction[0], cursor[1] - direction[1])
            c += 1

        return c - 1

    def __valid_index(self, index):
        return index[0] >= 0 and index[0] < self.board.shape[0] and index[1] >= 0 and index[1] < self.board.shape[1]

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
        return self.winner != 0 or np.sum(1-self.get_occupied()) == 0

    def winner(self):
        return self.winner

    def winner_from_current_players_perspective(self):
        return -1 * self.winner_from_last_players_perspective()

    def winner_from_last_players_perspective(self):
        if self.last_player == 0:
            return self.winner
        else:
            return -1 * self.winner

