
class MonteCarloTreeSearch:
    def __init__(self, game, model, max_depth = 5, max_branching = 15):
        self.game = game
        self.model = model
        self.max_depth = max_depth
        self.max_branching = max_branching

    def search(self):
        actions = self.model.most_probable_actions(self.game, self.max_branching)
        result = {
            'action': actions[0],
            'outcome': -1
        }

        for action in actions:
            try:
                self.game.take_action(action)

                new_outcome = self.__get_outcome(0, result['outcome'], -1)

                if new_outcome > result['outcome']:
                    result = {
                        'action': action,
                        'outcome': new_outcome
                    }

                self.game.revert()

                if result['outcome'] == 1:
                    break
            except Exception as e:
                print('Exception', e)
                raise e

        return result


    def __search_outcome(self, depth, alpha, beta):
        if depth > self.max_depth:
            return self.model.predicted_outcome(self.game)

        outcome = -1
        actions = self.model.most_probable_actions(self.game, self.max_branching)
        for action in actions:
            try:
                self.game.take_action(action)

                new_outcome = self.__get_outcome(depth, outcome, beta)

                if new_outcome > outcome:
                    outcome = new_outcome

                self.game.revert()

                if outcome >= -beta:
                    break
            except Exception as e:
                print('Exception', e)
                raise e

        return outcome

    def __get_outcome(self, depth, alpha, beta):
        if self.game.game_over():
            return self.game.winner_from_last_players_perspective()

        return -1 * self.__search_outcome(depth+1, beta, alpha)
