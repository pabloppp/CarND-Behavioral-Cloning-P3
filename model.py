import csv
import random

import cv2
import sys
import matplotlib
import numpy as np
import sklearn
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, AveragePooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load data
print("Loading data...")

if len(sys.argv) > 1:
    base_data_path = sys.argv[1]
else:
    base_data_path = "../data/"

if len(sys.argv) > 2:
    output_path = sys.argv[2]
else:
    output_path = "./"

data_path = base_data_path + 'custom_simple_base/'

lines = []
with open(data_path + 'driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
batch_size = 32


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steer_measurements = []

            for line in batch_samples:
                for i in range(0, 3):
                    image_source_path = line[i]
                    filename = image_source_path.split('/')[-1]
                    current_path = data_path + 'IMG/' + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)

                steer_correction = 0.18
                steer_measurement = float(line[3])
                steer_measurements.append(steer_measurement)
                steer_measurements.append(steer_measurement + steer_correction)
                steer_measurements.append(steer_measurement - steer_correction)

            # data augmentation
            augmented_images, augmented_steer_measurements = [], []
            for image, steer_measurement in zip(images, steer_measurements):
                augmented_images.append(image)
                augmented_steer_measurements.append(steer_measurement)
                augmented_images.append(np.fliplr(image))
                augmented_steer_measurements.append(-steer_measurement)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_steer_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# lambdas
def min_max(x):
    x_min = K.min(x, axis=[1, 2, 3], keepdims=True)
    x_max = K.max(x, axis=[1, 2, 3], keepdims=True)
    return (x - x_min) / (x_max - x_min) - 0.5


# model
model = Sequential()
model.add(Cropping2D(cropping=((65, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(AveragePooling2D(pool_size=(1, 2)))
model.add(Lambda(min_max))
model.add(Convolution2D(24, (8, 8), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (8, 8), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(70))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

checkpoint = ModelCheckpoint(output_path + 'model.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=2 * len(train_samples) / batch_size,
                                     validation_data=validation_generator,
                                     validation_steps=2 * len(validation_samples) / batch_size,
                                     epochs=30, callbacks=[checkpoint], verbose=2)  # 1 gives more data than 2

# model.save(output_path + 'model.h5')

# plot loss
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
plt.savefig(output_path + 'loss.png')
