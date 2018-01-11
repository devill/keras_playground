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
        return self.__search_outcomes(games)

    def __search_outcomes(self, games, depth = 0):
        boards = []
        tasks = []

        for game in games:
            if game.game_over():
                tasks.append({'task':'game_over', 'result': game.winner_from_current_players_perspective()})
            else:
                board = game.get_state_for_current_player()
                tasks.append({'task':'take_action', 'board_index': len(boards), 'game': game})
                boards.append(board)

        if depth > self.max_depth:
            outcomes = self.model.get_predicted_outcomes(np.array(boards))

            best_outcomes =[]

            for task in tasks:
                if task['task'] == 'take_action':
                    best_outcomes.append(-1*outcomes[task['board_index']])
                else:
                    best_outcomes.append(-1*task['result'])

            return (best_outcomes,[None]*len(best_outcomes))

        else:
            maps = self.model.get_probability_maps(np.array(boards))

            next_games = []

            for task in tasks:
                if task['task'] == 'take_action':
                    game = task['game']
                    pmap = maps[task['board_index']]

                    pmap = np.multiply(pmap, 1-game.get_occupied())
                    pmap = pmap / np.sum(pmap)

                    actions = choice2d(pmap, self.max_branching)
                    task['actions'] = actions
                    task['range_from'] = len(next_games)

                    for action in actions:
                        g = deepcopy(task['game'])
                        g.take_action(action)
                        next_games.append(g)

                    task['range_to'] = len(next_games)

            print(len(next_games))
            outcomes, actions = self.__search_outcomes(next_games, depth + 1)
            print(len(outcomes), len(next_games))

            best_outcomes = []
            best_actions = []

            for task in tasks:
                if task['task'] == 'take_action':
                    #print(task['range_from'], task['range_to'], len(outcomes), outcomes[task['range_from']:task['range_to']])
                    best_outcomes.append(-1*max(outcomes[task['range_from']:task['range_to']]))
                    best_actions.append(task['actions'][np.argmax(outcomes[task['range_from']:task['range_to']])])
                else:
                    best_outcomes.append(task['result'])
                    best_actions.append(None)

            return (best_outcomes, best_actions)






