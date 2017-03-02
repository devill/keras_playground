import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

base_version = sys.argv[1]
weights_filename = sys.argv[2]

DATA_PATH = '/data/dogscats/'

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
    model.add(Dense(2, activation='softmax'))
    
    return model

model = BuildVGG()
model.load_weights(weights_filename)

model.compile(optimizer=Adam(),loss='categorical_crossentropy', metrics=['accuracy'])

valid_batches = image.ImageDataGenerator().flow_from_directory(
    DATA_PATH +'valid',
    target_size=(224, 224),
    class_mode='categorical',
    shuffle=True,
    batch_size=16
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
    class_mode='categorical',
    shuffle=True,
    batch_size=32
)

#print(model.evaluate_generator(valid_batches,2000))

for i in range(3):
    model.optimizer.lr = 0.1**(5+i)
    result = model.fit_generator(train_batches,
                        samples_per_epoch=train_batches.nb_sample,
                        nb_epoch=1,
                        validation_data=valid_batches,
                        nb_val_samples=valid_batches.nb_sample
    )

    
stats = '_'.join(['%s-%.5f' % (key,values[-1]) for (key,values) in sorted(result.history.items())])
model.save_weights('/data/trained_models/dogscats/ensamble/full_trained_models/v'+base_version+'.'+str(i)+'_'+stats+'.h5')
    
