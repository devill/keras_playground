from utils import *
import os
from gomoku_with_group_map import GomokuWithGroupMap
from hand_crafted_gomoku_model import HandCraftedGomokuModel
from monte_carlo_tree_search import MonteCarloTreeSearch
import cProfile
import sys
import hashlib
import numpy as np
import random

shape=(19,19)

def play(game_index = None):
    game = GomokuWithGroupMap(shape)
    tree_search = MonteCarloTreeSearch(game, HandCraftedGomokuModel(), 2, 10)
    results = []

    os.system('clear')
    game.char_draw()
    while not game.game_over():
        result = tree_search.search()

        if random.random() < 0.2:
            real_action = result['action']
        else:
            valid_moves = 1-np.sum(game.board[:,:,:2], axis=2)
            real_action = choice2d(valid_moves/np.sum(valid_moves), 1)[0]

        result['real_action'] = real_action

        results.append(result)
        game.take_action(real_action)

        os.system('clear')
        if game_index:
            print(str(game_index) + " Outcome: " + str(result['outcome']) + "  Action: " + str(result['action']))
        else:
            print("Outcome: " + str(result['outcome']) + "  Action: " + str(result['action']))

        game.char_draw()

    return results

arg = sys.argv[1] if len(sys.argv) > 1 else ''

if arg == 'profile':
    cProfile.run('play()')
elif arg == 'generate':
    for i in range(int(sys.argv[2])):
        results = play(i)

        history = "\n".join(list(map(lambda x: str(x['outcome']) +","+str(x['action'][0])+","+str(x['action'][1])+","+str(x['real_action'][0])+","+str(x['real_action'][1]), results)))
        m = hashlib.md5()
        m.update(history.encode('utf-8'))
        h = m.hexdigest()

        with open('/data/gomoku_alpha_zero/2_10_with_random/'+h+'.csv', 'w') as f:
            f.write(history)

else:
    results = play()

    for r in results:
        print("Outcome: " + str(r['outcome']) + "  Action: " + str(r['action']))
