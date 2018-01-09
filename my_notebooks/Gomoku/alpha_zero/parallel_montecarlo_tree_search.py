import numpy as np
from copy import deepcopy

def choice2d(pmap, count = 10):
    shape = pmap.shape
    indices = np.transpose(np.indices(shape), axes=(1,2,0)).reshape((shape[0]*shape[1],2))
    choice_indices = np.random.choice(len(indices), count, p=pmap.reshape(shape[0]*shape[1]))
    return list(set(map(lambda x: tuple(x), indices[choice_indices].tolist())))

class ParallelMonteCarloTreeSearch:
    def __init__(self, model, max_depth = 5, max_branching = 15):
        self.model = model
        self.max_depth = max_depth
        self.max_branching = max_branching

    def search(self, games):


        pass

    def __search_outcomes(self, games, depth = 0):
        if depth > self.max_depth:
            outcomes = []

            #TODO

        else:
            boards = []
            tasks = []

            for game in games:
                if game.game_over():
                    tasks.append({'task':'game_over', 'result': game.winner_from_current_players_perspective()})
                else:
                    board = game.get_state_for_current_player()
                    tasks.append({'task':'take_action', 'board_index': len(board), 'game': game})
                    boards.append(board)

            maps = self.model.get_probability_maps(np.array(boards))

            next_games = []

            for task in tasks:
                if task['task'] == 'take_action':
                    pmap = maps[task['board_index']]
                    actions = choice2d(pmap, self.max_branching)
                    task['range_from'] = len(next_games)

                    for action in actions:
                        g = deepcopy(task['game'])
                        g.take_action(action)
                        next_games.append(g)

                    task['range_to'] = len(next_games)

            outcomes = self.__search_outcomes(next_games, depth + 1)

            results = []

            for task in tasks:
                if task['task'] == 'take_action':
                    results.append(-1*max(outcomes[task['range_from']:task['range_to']]))
                else:
                    results.append(task['result'])

            return results






