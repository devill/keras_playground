import numpy as np
from numpy.random import random, permutation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import itertools
import bcolz


def TrainFCLayers(version, epochs):
    valid_features = bcolz.open('/data/precalc_conv_layers_valid.bc')[:]
    train_features = bcolz.open('/data/precalc_conv_layers_train.bc')[:]
    
    valid_batches = image.ImageDataGenerator().flow_from_directory(
        '/data/dogscats/valid', 
        target_size=(224,224),
        class_mode='categorical', 
        shuffle=False, 
        batch_size=32
    )

    train_batches = image.ImageDataGenerator().flow_from_directory(
        '/data/dogscats/train', 
        target_size=(224,224),
        class_mode='categorical', 
        shuffle=False, 
        batch_size=32
    )

    
    fc_model = Sequential([
        Dense(4096, input_shape=(25088,), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(2, activation='softmax'),
    ])

    fc_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    for ep in range(epochs):
        result = fc_model.fit(
            train_features, 
            to_categorical(train_batches.classes), 
            validation_data=(valid_features, to_categorical(valid_batches.classes)),
            nb_epoch=1
        )
        stats = '_'.join(['%s-%.5f' % (key,values[-1]) for (key,values) in sorted(result.history.items())])
        fc_model.save_weights('/data/trained_models/dogscats/ensamble/fc_ensamble_' + str(version) + '.' + str(ep) + '_' + stats + '.h5')
        fc_model.optimizer.lr = fc_model.optimizer.lr/10

for i in range(9,10):
    TrainFCLayers(i,2)
