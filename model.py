import tensorflow as tf
import keras
import numpy as np
import csv
import cv2

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


from os.path import exists

raw = []
camera_correction = 0.2 
dataset = 'new_data/'
with open(dataset+'driving_log.csv', 'r') as f:
    csv_reader = csv.reader(f)
    #header = next(csv_reader)
    for line in csv_reader:
        steering = float(line[3])
        # path = dataset + line[0]
        # if exists(path):
        #     raw.append((path, steering))
        # path = dataset + line[1][1:]
        # if exists(path):
        #     raw.append((path, steering-camera_correction))
        # path = dataset + line[2][1:]
        # if exists(path):
        #     raw.append((path, steering+camera_correction))
        raw.append((line[0], steering))
        raw.append((line[1], steering+camera_correction))
        raw.append((line[2], steering-camera_correction))

from sklearn.model_selection import train_test_split
train_raw, validation_raw = train_test_split(raw, test_size=0.2)



from sklearn.utils import shuffle

def sample_generator(raw, batch_size=32):
    N = len(raw)
    while True:
        raw = shuffle(raw)
        for offset in range(0, N, batch_size):
            batch = raw[offset:offset+batch_size]
            imgs = []
            steerings = []
            for path, steering in batch:
                img = cv2.imread(path)
                imgs.append(img)
                steerings.append(steering)
                imgs.append(cv2.flip(img, 1))
                steerings.append(-steering)
            
            yield np.array(imgs), np.array(steerings)

train_sample_generator = sample_generator(train_raw, 32)
validation_sample_generator = sample_generator(validation_raw, 32)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def myModel():
    model = Sequential()

    model = addPreProcess(model)

    model.add(Conv2D(24,5, strides=(2,2), activation='relu'))
    model.add(Conv2D(36,5, strides=(2,2), activation='relu'))
    model.add(Conv2D(48,5, strides=(2,2), activation='relu'))
    model.add(Conv2D(64,3, activation='relu'))
    model.add(Conv2D(64,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    return model

def addPreProcess(model):
    model.add(Lambda(lambda x: (x-128)/255.0, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

model = myModel()

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_sample_generator, \
                                     steps_per_epoch=np.ceil(len(train_raw)/16), \
                                     validation_data=validation_sample_generator, \
                                     validation_steps=np.ceil(len(validation_raw)/16), \
                                     epochs=3, \
                                     verbose=1)

model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])