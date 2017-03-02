import numpy as np
from numpy.random import random, permutation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from PIL import Image
import itertools
import sys

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr

def ConvBlock(model, layers, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))
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

def RemoveFCLayers(model):
    while type(model.layers[-1]).__name__ != 'Flatten':
        model.pop()

def AddFCLayers(model):
    model.add(Dense(4096, input_shape=(25088,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

def CreateConvModelFromFCWeightFile(in_file):
    fc_model = Sequential()
    AddFCLayers(fc_model)

    fc_model.load_weights(in_file)
    fc_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    cd_model = BuildVGG()
    RemoveFCLayers(cd_model)
    AddFCLayers(cd_model)
    
    for i in range(7):
        cd_model.layers[-(1+i)].set_weights(fc_model.layers[-(i+1)].get_weights())
    
    cd_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])        
    return cd_model

valid_batches = image.ImageDataGenerator().flow_from_directory(
    '/data/dogscats/valid', 
    target_size=(224,224),
    class_mode='categorical', 
    shuffle=False, 
    batch_size=32
)

in_file = sys.argv[1]
out_file = sys.argv[2]

print(in_file)
model = CreateConvModelFromFCWeightFile(in_file)
print(model.evaluate_generator(valid_batches,2000))

model.save_weights(out_file)