from gomoku import Gomoku
from learned_gomoku_model import LearnedGomokuModel
from keras.models import load_model
from parallel_monte_carlo_tree_search import ParallelMonteCarloTreeSearch
import hashlib
import sys

shape = (19,19)
batch_size = 16
model = load_model('/data/trained_models/gomoku_alpha_zero/gomoku_alpha_zero_resnet_full_model_v5.h5')

games = [Gomoku(shape) for _ in range(batch_size)]
histories = [[] for _ in range(batch_size)]

lmodel = LearnedGomokuModel(model)
tree_search = ParallelMonteCarloTreeSearch(LearnedGomokuModel(model), 2, 6)
results = []

def games_over(games):
    for game in games:
        if not game.game_over():
            return False
    return True

while not games_over(games):
    outcomes, actions = tree_search.search(games)
    results.append({'outcome':outcomes[0], 'action':actions[0]})
    for i in range(len(games)):
        if not games[i].game_over():
            games[i].take_action(actions[i])
            histories[i].append({
                'outcome':outcomes[i],
                'action':actions[i]
            })
    print('.', end='')
    sys.stdout.flush()

print('')

for history in histories:
    file_content = "\n".join(list(map(lambda x: str(x['outcome']) +","+str(x['action'][0])+","+str(x['action'][1]), history)))
    m = hashlib.md5()
    m.update(file_content.encode('utf-8'))
    h = m.hexdigest()

    with open('/data/gomoku_alpha_zero/reinforcement_1/'+h+'.csv', 'w') as f:
        f.write(file_content)
