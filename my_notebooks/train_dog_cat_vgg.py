import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.optimizers import Adam

DATA_PATH = '/data/dogscats/'
INITIAL_WEIGHTS_FILE = '/data/trained_models/untrained_dog_cat_vgg.h5'
EVENTUAL_WEIGHTS_FILE = '/data/trained_models/trained_dog_cat_vgg_1.h5'

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

def train_model(trained_model, epoch = 1):
    valid_batches = image.ImageDataGenerator().flow_from_directory(
        DATA_PATH +'valid',
        target_size=(224, 224),
        class_mode='binary',
        shuffle=False,
        batch_size=8
    )
    train_batches = image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        dim_ordering='tf'
    ).flow_from_directory(
        DATA_PATH+'train',
        target_size=(224, 224),
        class_mode='binary',
        shuffle=True,
        batch_size=16
    )
    trained_model.fit_generator(train_batches,
                        samples_per_epoch=train_batches.nb_sample,
                        nb_epoch=epoch,
                        validation_data=valid_batches,
                        nb_val_samples=valid_batches.nb_sample
    )


def build_model():
    seq_model = Sequential()
    seq_model.add(Lambda(vgg_preprocess, input_shape=(224, 224, 3)))

    ConvBlock(seq_model, 2, 64)
    ConvBlock(seq_model, 2, 128)
    ConvBlock(seq_model, 3, 256)
    ConvBlock(seq_model, 3, 512)
    ConvBlock(seq_model, 3, 512)

    seq_model.add(Flatten())

    FCBlock(seq_model)
    FCBlock(seq_model)

    seq_model.add(Dense(1, activation='sigmoid'))

    seq_model.load_weights(INITIAL_WEIGHTS_FILE)

    seq_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return seq_model


model = build_model()

layers = model.layers
for layer in layers: layer.trainable=False
layers[-1].trainable=True

# layers = model.layers
# first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
# for layer in layers[first_dense_idx:]: layer.trainable=True

train_model(model, 1)

model.save(EVENTUAL_WEIGHTS_FILE)