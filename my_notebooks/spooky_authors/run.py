import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

import csv

raw_train_data = []
raw_train_labels = []

with open('/data/spooky_author/train_ascii.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for lid, text, author in reader:
        raw_train_data.append(text)
        raw_train_labels.append(author)
        
        
chars = ['@'] + sorted(list(set(''.join(raw_train_data))))
vocab_size = len(chars)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

max_len = 400

tdata = np.array([[char_indices[c] for c in '@'*max_len+text][-max_len:] for text in raw_train_data])

authors = sorted(list(set(raw_train_labels)))
author_indices = dict((a, i) for i, a in enumerate(authors))
indices_autor = dict((i, a) for i, a in enumerate(authors))

tlabels = to_categorical(list(map(lambda l: author_indices[l], raw_train_labels)))

number_of_classes = len(authors)

from sklearn.model_selection import train_test_split

train_data, valid_data, train_labels, valid_labels  = train_test_split(tdata, tlabels, test_size=0.05)

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam

n_fac = 32 
bs = 128
n_hidden=32

model=Sequential([
        Embedding(vocab_size, n_fac, batch_input_shape=(bs,max_len)),
        BatchNormalization(),
        LSTM(n_hidden, return_sequences=True),
        LSTM(n_hidden),
        Dense(number_of_classes, activation='softmax')
    ])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Nadam(), metrics=['accuracy'])

model.fit(train_data, train_labels, validation_data=(valid_data, valid_labels), batch_size=bs)

model.save_weights('/data/trained_models/spooky_v1.h5')
