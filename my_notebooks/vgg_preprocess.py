import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.optimizers import Adam
import json
import math


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


def generate(type):
    global batch_size, batches, batch_count, f, batch_index, imgs, labels, preds, i
    batch_size = 32
    batches = image.ImageDataGenerator().flow_from_directory(
        '../example_data/dogscats/' + type,
        target_size=(224, 224),
        class_mode='binary',
        shuffle=False,
        batch_size=batch_size
    )
    batch_count = math.ceil(batches.nb_sample / batch_size)
    with open('/data/dogscats/precalculated_vgg_' + type + '.data', 'w') as f:
        for batch_index in range(batch_count):
            imgs, labels = next(batches)
            preds = model.predict(imgs)
            for i in range(len(labels)):
                json.dump([int(labels[i]), preds[i].tolist()], f)
                f.write('\n')
            print("Batch " + str(batch_index) + "/" + str(batch_count) + " done")


generate('train')
generate('valid')
