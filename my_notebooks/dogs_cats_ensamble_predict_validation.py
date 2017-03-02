import numpy as np
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.optimizers import Adam
import sys

weights_filename = sys.argv[1]

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
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
def BuildVGG(filename):
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
    model.add(Dense(2, activation='softmax'))
    
    model.load_weights('/data/trained_models/dogscats/ensamble/full_trained_models/' + filename)
    model.compile(optimizer=Adam(lr=0.00000001),loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

test_batches = image.ImageDataGenerator().flow_from_directory(
    '/data/dogscats/valid', 
    target_size=(224,224),
    class_mode=None, 
    shuffle=False, 
    batch_size=128
)

def filename_to_id(filename):
    lead = 100000 if filename[:3] == 'cat' else 200000
    
    return lead + int(filename[9:-4])

ids = [filename_to_id(s) for s in test_batches.filenames]

model = BuildVGG(weights_filename)

results = model.predict_generator(test_batches,test_batches.nb_sample)
data = [[ids[i], results[i][1]] for i in range(len(ids))]

np.savetxt('/data/dogscats/ensamble_results/'+weights_filename[:2]+'_validation.csv', data ,fmt="%d,%.5f", header='id,label')

