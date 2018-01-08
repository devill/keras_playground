import numpy as np
from gomoku import Gomoku

class GomokuWithGroupMap(Gomoku):
    def reset(self):
        super().reset()
        self.group_map = np.zeros((2, self.shape[0], self.shape[1], 4))

    def get_group_map(self):
        return self.group_map

    def take_action(self, action):
        super().take_action(action)
        self.__update_group_map(action)

    def revert(self):
        action = self.action_stack[-1]
        self.__update_group_map(action, -1)
        super().revert()

    def __update_group_map(self, action, direction = 1):
        gmap = self.group_map[self.last_player]

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