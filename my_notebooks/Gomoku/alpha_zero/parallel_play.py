from gomoku import Gomoku
from learned_gomoku_model import LearnedGomokuModel
from keras.models import load_model
from parallel_monte_carlo_tree_search import ParallelMonteCarloTreeSearch
import hashlib
import numpy as np
import os
import random

shape = (19,19)
batch_size = 16
model = load_model('/data/trained_models/gomoku_alpha_zero/gomoku_alpha_zero_resnet_full_model_v6.h5')

def action_to_onehot(action):
    result = np.zeros(shape)
    result[action] = 1
    return result

def games_over(games):
    for game in games:
        if not game.game_over():
            return False
    return True

def board_augmentation(inp, out):
    sym = random.choice([' ','|','\\'])
    if sym == '|':
        inp = np.flip(inp,axis=0)
        out = np.flip(out,axis=0)
    elif sym =='\\':
        inp = np.transpose(inp, axes=(1,0,2))
        out = np.transpose(out)

    k = random.randint(0,3)
    return np.rot90(inp,k=k, axes=(0,1)), np.rot90(out,k=k, axes=(0,1))

def display_games(games):
    ascii_arts = list(map(lambda g: g.char_draw_as_string().split("\n"), games))

    display = ''
    for l in range(len(ascii_arts[0])):
        for i in range(len(ascii_arts)):
            display += ascii_arts[i][l] + '   '
        display += "\n"

    print(display)

iterations = 0

while True:
    games = [Gomoku(shape) for _ in range(batch_size)]
    histories = [[] for _ in range(batch_size)]

    lmodel = LearnedGomokuModel(model)
    tree_search = ParallelMonteCarloTreeSearch(LearnedGomokuModel(model), 2, 6)
    results = []



    os.system('clear')
    display_games(games[0:4])
    display_games(games[4:8])
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

        os.system('clear')
        display_games(games[0:4])
        display_games(games[4:8])


    train_boards = []
    train_scores = []
    train_actions = []

    base = np.ones(shape)
    base[games[0].get_middle()] += 1

    for history in histories:

        board = np.stack((np.zeros(shape), np.zeros(shape),np.copy(base)), axis=2)
        player = 0

        for x in history:
            original = np.copy(board)
            action = action_to_onehot((x['action'][0], x['action'][1]))

            original, action = board_augmentation(original, action)

            train_boards.append(original)
            train_scores.append(x['outcome'])
            train_actions.append(action.flatten())

            board[x['action'][0], x['action'][1], player] = 1
            player = 1 - player

    model.fit(np.array(train_boards), [np.array(train_actions),np.array(train_scores)], shuffle=True, batch_size=128)


    for history in histories:
        file_content = "\n".join(list(map(lambda x: str(x['outcome']) +","+str(x['action'][0])+","+str(x['action'][1]), history)))
        m = hashlib.md5()
        m.update(file_content.encode('utf-8'))
        h = m.hexdigest()

        with open('/data/gomoku_alpha_zero/reinforcement_3/'+h+'.csv', 'w') as f:
            f.write(file_content)


    iterations += 1
    if iterations % 20 == 0:
        model.save('/data/trained_models/gomoku_alpha_zero/gomoku_alpha_zero_resnet_full_model_v7_'+str(iterations)+'.h5')
        model.save_weights('/data/trained_models/gomoku_alpha_zero/gomoku_alpha_zero_resnet_weights_v7_'+str(iterations)+'.h5')