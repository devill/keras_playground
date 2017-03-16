import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.optimizers import Nadam
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

from keras import backend as K
from keras.datasets import mnist as mnist_dataset

class MNISTImages:
    def __init__(self):
        img_rows = img_cols = 28
        
        (X_train, y_train), (X_test, y_test) = mnist_dataset.load_data()

        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
            
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
            
        self.train_input = X_train
        self.train_labels = self.to_categorical(y_train)
        
        self.test_input = X_test
        self.test_labels = self.to_categorical(y_test)
        
    def to_categorical(self,y):
        y = np.array(y, dtype='int').ravel()
        nb_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, nb_classes))
        categorical[np.arange(n), y] = 1
        return categorical
        

def getModel():
    return Sequential([
        BatchNormalization(input_shape=(28,28,1), axis=3),
        Convolution2D(10,3,3, border_mode='same', activation='relu'),
        Convolution2D(10,3,3, border_mode='same', activation='relu'),
        BatchNormalization(axis=3),
        Dropout(0.1),
        MaxPooling2D(pool_size=(2,2)),
        Convolution2D(20,3,3, border_mode='same', activation='relu'),
        Dropout(0.2),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.45),
        Dense(10),
        Activation('softmax')
    ])


model = getModel()
opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.006)
model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])

mnist = MNISTImages()

train_generator = image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.1,
    dim_ordering='tf'
).flow(mnist.train_input, mnist.train_labels, batch_size=512)

test_generator = image.ImageDataGenerator().flow(mnist.test_input, mnist.test_labels)

model.load_weights('/data/trained_models/mnist/v0.2.h5')
model.optimizer.lr = 0.00000002

model.fit_generator(
        train_generator, 
        samples_per_epoch=len(mnist.train_labels),
        nb_epoch=4, 
        validation_data=test_generator, 
        nb_val_samples=len(mnist.test_labels)
    )

print(model.evaluate(mnist.test_input, mnist.test_labels, verbose=0))

model.save_weights('/data/trained_models/mnist/v0.3.h5')