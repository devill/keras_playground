import numpy as np
from numpy.random import random, permutation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import itertools

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr

def ConvBlock(model, layers, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
def FCBlock(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    
def BuildVGG():
    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(224,224,3)))
    ConvBlock(model, 2, 64)
    ConvBlock(model, 2, 128)
    ConvBlock(model, 3, 256)
    ConvBlock(model, 3, 512)
    ConvBlock(model, 3, 512)

    model.add(Flatten())
    FCBlock(model)
    FCBlock(model)
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('/data/trained_models/vgg16_tf.h5')
    model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = BuildVGG()

# find the index of the flatten layer
original_layers = model.layers
flatten = [x for x in original_layers if x.name == 'flatten_1'][0]
flatten_idx = original_layers.index(flatten)

conv_layers = original_layers[:flatten_idx+1]
fc_layers = original_layers[flatten_idx+1:]

conv_model = Sequential(conv_layers)

valid_batches = image.ImageDataGenerator().flow_from_directory(
    '/data/cervical/valid', 
    target_size=(224,224),
    class_mode='categorical', 
    shuffle=False, 
    batch_size=32
)

train_batches = image.ImageDataGenerator().flow_from_directory(
    '/data/cervical/train', 
    target_size=(224,224),
    class_mode='categorical', 
    shuffle=False, 
    batch_size=32
)

valid_features = conv_model.predict_generator(valid_batches, 32, verbose=1)
