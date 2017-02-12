import numpy as np
from PIL import Image
from keras.models import load_model
from keras import backend as keras_backend

model = load_model('/data/trained_models/gol_learnt.h5')

im = Image.open('/data/gol_img/small.jpg')
raw_data = np.asarray(im, dtype='float32')/255

if keras_backend.image_dim_ordering() == 'th':
    data = raw_data.reshape(1, 1, 64, 64)
else:
    data = raw_data.reshape(1, 64, 64, 1)

for i in range(40):
    data = np.round(model.predict(data))
    # data = model.predict(data)
    raw_result = data.reshape(64, 64) * 255
    Image.fromarray(raw_result).convert('RGB').save('/data/gol_img/result_{0:0=2d}.jpg'.format(i))
