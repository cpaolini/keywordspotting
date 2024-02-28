# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:04:59 2023

@author: Aarti
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile
import os
import time
import pickle
import random
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

#for CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from keras.layers import LSTM, Bidirectional
import keras.backend as K

#for MLP model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization ,Activation, Flatten, Permute, Reshape, Lambda
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.utils import np_utils
from sklearn import metrics
from datetime import datetime

from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch

from tensorflow.keras import Model, Input
from tcn import TCN, compiled_tcn


X = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/X.npy')
Y = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/Y.npy')
  
X_train,X_valtest,Y_train,Y_valtest = train_test_split(X,Y,test_size=0.2,random_state=37)
X_val,X_test,Y_val,Y_test = train_test_split(X_valtest,Y_valtest,test_size=0.5,random_state=37)

print("shape of X_train is:",X_train.shape)
print("shape of X_Val is:",X_val.shape)
print("shape of X_Test is:", X_test.shape)
print("shape of Y_train is:",Y_train.shape)
print("shape of Y_Val is:",Y_val.shape)
print("shape of Y_Test is:", Y_test.shape)

num_rows = X_train.shape[1]
num_columns = X_train.shape[2]
num_channels = 1
num_labels = Y_train.shape[1]
  # num_labels = Ymlb.shape[1]
  
X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)

model = Sequential()
model.add(Conv2D(64, kernel_size=5,input_shape=(num_rows, num_columns, num_channels), activation='relu')) 
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(96,(3,3),activation='relu')) 
model.add(Conv2D(384,(3,3),activation='relu'))
model.add(Conv2D(384,(3,3),activation='relu'))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(num_labels, activation='softmax'))

opt = Adam(learning_rate=0.00019027533034000736)

model.summary()

import visualkeras
from PIL import ImageFont

font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
visualkeras.layered_view(model, legend=True, font=font).show()

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='CNN_model_plot.png', show_shapes=True, show_layer_names=True)



# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

# score = model.evaluate(X_test, Y_test, verbose=1)
# accuracy = 100*score[1]

# num_epochs = 50
# num_batch_size = 32

# model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), verbose=1)

# # Evaluating the model on the training and testing set
# score = model.evaluate(X_train, Y_train, verbose=0)
# print("Training Accuracy: ", score[1])

# score = model.evaluate(X_test, Y_test, verbose=0)
# print("Testing Accuracy: ", score[1])

# #Predictions
# Y_pred = model.predict(X_test, batch_size=32, verbose=1)
# np.set_printoptions(precision=5, suppress=True)
# print(Y_pred)



