from keras.models import Sequential
from keras.layers import Convolution2D, Activation
from keras import backend as keras_backend
import numpy as np
from scipy import signal

nof_generated_train_example = 1024
nof_generated_test_example = 64
input_img_rows = 64
input_img_cols = 64


def generate_input_data(nof_examples, witdth, height):
    return np.random.randint(0,2, size=(nof_examples, witdth, height))

def generate_neighbourhood_matrix(input):
    filter = np.array([[1,1,1],[1,0,1],[1,1,1]])
    result = np.zeros(input.shape)

    for i in range(input.shape[0]):
        result[i] = signal.convolve2d(input[i], filter, mode='same')

    return result

def generate_result_data(input):
    neighbours = generate_neighbourhood_matrix(input)
    survivors_with_two_neighbours = (np.ones(input.shape) * 2 == neighbours).astype(int) * input
    cells_with_three_neighbours = (np.ones(input.shape) * 3 == neighbours).astype(int)
    return survivors_with_two_neighbours + cells_with_three_neighbours


input_train  = generate_input_data(nof_generated_train_example, input_img_rows, input_img_rows)
result_train = generate_result_data(input_train)

input_test  = generate_input_data(nof_generated_test_example, input_img_rows, input_img_rows)
result_test = generate_result_data(input_test)

if keras_backend.image_dim_ordering() == 'th':
    input_train = input_train.reshape(input_train.shape[0], 1, input_img_rows, input_img_cols)
    input_test = input_test.reshape(input_test.shape[0], 1, input_img_rows, input_img_cols)
    result_train = result_train.reshape(result_train.shape[0], 1, input_img_rows, input_img_cols)
    result_test = result_test.reshape(result_test.shape[0], 1, input_img_rows, input_img_cols)
    input_shape = (1, input_img_rows, input_img_cols)
else:
    input_train = input_train.reshape(input_train.shape[0], input_img_rows, input_img_cols, 1)
    input_test = input_test.reshape(input_test.shape[0], input_img_rows, input_img_cols, 1)
    result_train = result_train.reshape(result_train.shape[0], input_img_rows, input_img_cols, 1)
    result_test = result_test.reshape(result_test.shape[0], input_img_rows, input_img_cols, 1)
    input_shape = (input_img_rows, input_img_cols, 1)

model = Sequential()

model.add(Convolution2D(10, 3, 3, border_mode='same', input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(8, 1, 1, border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(6, 1, 1, border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(4, 1, 1, border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(1, 1, 1, border_mode='valid'))
model.add(Activation('sigmoid'))

# weights = model.get_weights()
#
# with open('gol_model_manual_weights.json') as data_file:
#     manual_model = json.load(data_file)
#
# for i in range(len(manual_model)):
#     weights[i] = np.array(manual_model[i], dtype='float32')
#
# model.set_weights(weights)

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(input_train, result_train, batch_size=256, nb_epoch=1024,
          verbose=1, validation_data=(input_test, result_test))

score = model.evaluate(input_test, result_test, verbose=0)

print('Test score:', score[0])

model.save('/data/trained_models/gol.h5')
