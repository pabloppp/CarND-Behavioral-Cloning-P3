import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

# load data
data_path = '../data/custom_data/'
lines = []
with open(data_path + 'driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

images = []
stir_measurements = []
for line in lines:
    image_source_path = line[0]
    filename = image_source_path.split('/')[-1]
    current_path = data_path + 'IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)

    stir_measurement = float(line[3])
    stir_measurements.append(stir_measurement)


def min_max(x):
    x_min = K.min(x, axis=[1, 2, 3], keepdims=True)
    x_max = K.max(x, axis=[1, 2, 3], keepdims=True)
    return (x - x_min) / (x_max - x_min) - 0.5


def yuv_conversion(x):
    import tensorflow as tf
    return tf.image.rgb_to_yuv(x)


# model

X_train = np.array(images)
y_train = np.array(stir_measurements)

model = Sequential()
model.add(Lambda(yuv_conversion, input_shape=(160, 320, 3)))
model.add(Lambda(min_max))
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(84, activation="relu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=20)

model.save('model.h5')
