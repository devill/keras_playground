import os
from gomoku import Gomoku
from hand_crafted_gomoku_model import HandCraftedGomokuModel
from monte_carlo_tree_search import MonteCarloTreeSearch
import cProfile
import sys
import hashlib

shape=(19,19)

def play():
    game = Gomoku(shape)
    tree_search = MonteCarloTreeSearch(game, HandCraftedGomokuModel(), 2, 10)
    results = []

    os.system('clear')
    game.char_draw()
    while not game.game_over():
        result = tree_search.search()
        results.append(result)
        game.take_action(result['action'])

        os.system('clear')
        print(str(i) + " Outcome: " + str(result['outcome']) + "  Action: " + str(result['action']))
        game.char_draw()

    return results

arg = sys.argv[1] if len(sys.argv) > 1 else ''

if arg == 'profile':
    cProfile.run('play()')
elif arg == 'generate':
    for i in range(int(sys.argv[2])):
        results = play()

        history = "\n".join(list(map(lambda x: str(x['outcome']) +","+str(x['action'][0])+","+str(x['action'][1]), results)))
        m = hashlib.md5()
        m.update(history.encode('utf-8'))
        h = m.hexdigest()

        with open('/data/gomoku_alpha_zero/'+h+'.csv', 'w') as f:
            f.write(history)

else:
    results = play()

    for r in results:
        print("Outcome: " + str(r['outcome']) + "  Action: " + str(r['action']))