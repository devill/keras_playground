import math
import numpy as np
from scipy import signal
from utils import *


class HandCraftedGomokuModel:
    def predicted_outcome(self, game):
        gmap = game.get_group_map()

        points = self.__score(gmap[game.get_current_player_id()], gmap[game.get_last_player_id()])
        points -= self.__score(gmap[game.get_last_player_id()], gmap[game.get_current_player_id()])
        return math.tanh(points)

    def __score(self, me, enemy):
        enemy_mask = 1*(enemy == 0)
        return np.sum(np.vectorize(lambda x: pow(10,x)/100000)(np.multiply(me, enemy_mask)))

    def __search_mask(self, mask, me, enemy):
        enemy_mask = 1*(signal.convolve2d(enemy, mask, mode='same') == 0)
        my_positions = signal.convolve2d(me, mask, mode='same')
        my_open_positions = np.multiply(enemy_mask, my_positions)
        return np.sum(np.vectorize(lambda x: pow(4,x)/1024)(my_open_positions))

    def most_probable_actions(self, game, max_branching):
        if game.get_number_of_steps() == 0:
            return [game.get_middle()]

        pmap = self.get_probability_map(game)
        return list(set(choice2d(pmap , max_branching)))

    def get_probability_map(self, game):
        pmap = np.sum(np.sum(game.get_group_map(), axis=0), axis=2)
        pmap = np.multiply(pmap, 1-game.get_occupied())
        return pmap / np.sum(pmap)