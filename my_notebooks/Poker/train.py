import random
import glob
import json
import numpy as np

def get_card_as_one_hot(card):
    s = ['clubs','hearts','spades','diamonds'].index(card['suit'])
    v = ([ str(i) for i in range(2,11) ] + ['J','K','Q','A']).index(card['rank'])
    return v,s

def get_cards_as_one_hot(cards):
    one_hot = np.zeros((13,4))
    for c in cards:
        one_hot[get_card_as_one_hot(c)] = 1
    return one_hot

def get_all_cards_for_hand_for_player(hand, player_index):
    return np.array([
        get_cards_as_one_hot(hand['hole_cards'][player_index]), 
        get_cards_as_one_hot(hand['community_cards'][:3]),
        get_cards_as_one_hot(hand['community_cards'][3:4]),
        get_cards_as_one_hot(hand['community_cards'][4:5])
    ])

def get_a_winning_hand(hand):
    all_cards = get_all_cards_for_hand_for_player(hand, hand['winner'])
    split = random.randint(1,4)
    return np.concatenate((all_cards[:split],np.zeros((4-split, 13, 4))), axis=0)

def get_a_loosing_hand(hand):
    index = random.randint(0, len(hand['hole_cards'])-2)
    index = index if index < hand['winner'] else index
    all_cards = get_all_cards_for_hand_for_player(hand, index)
    split = random.randint(1,4)
    return np.concatenate((all_cards[:split],np.zeros((4-split, 13, 4))), axis=0)
    
    
inputs = []
outputs = []

for file in glob.glob("/data/poker/rawdata/*.json"):
    f = open(file, 'r')
    hands_in_game = json.loads(f.read())

    for hand in hands_in_game: 
        inputs.append(get_a_winning_hand(hand))
        outputs.append(1)
        inputs.append(get_a_loosing_hand(hand))
        outputs.append(0)
    
inputs = np.array(inputs)
outputs = np.array(outputs)



test_indices_set = set(([random.randint(0, 36081) for i in range(0,6500)]))

test_indices = sorted(list(test_indices_set))
train_indices = [i for j, i in enumerate(range(0,36081)) if j not in test_indices]

test_inputs = inputs[test_indices]
test_outputs = outputs[test_indices]
train_inputs = inputs[train_indices]
train_outputs = outputs[train_indices]



import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Nadam
from keras.preprocessing import image



model = Sequential([
    Convolution2D(10,4,5, border_mode='same', activation='relu', input_shape=(4, 13, 4)),
    Convolution2D(10,4,5, border_mode='same', activation='relu'),
    Flatten(),
    Dense(20, activation='relu'),
    Dense(1),
    Activation('sigmoid')
])


opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.006)
model.compile(optimizer=opt,loss='binary_crossentropy', metrics=['accuracy'])


train_generator = image.ImageDataGenerator().flow(train_inputs, train_outputs)
test_generator = image.ImageDataGenerator().flow(test_inputs, test_outputs)


model.fit_generator(
       train_generator, 
       samples_per_epoch=len(train_outputs),
       nb_epoch=24, 
       validation_data=test_generator, 
       nb_val_samples=len(test_outputs)
   )

model.save_weights('/data/trained_models/poker/winning_probability_by_hands_v3.h5')