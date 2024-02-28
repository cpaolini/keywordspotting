
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

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

# for ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

X_train = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/X.npy')
Y_train = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/Y.npy')

X_test = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/X_rhyming_word.npy')
Y_test = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/Y_rhyming_word.npy')
  
# X_train,X_valtest,Y_train,Y_valtest = train_test_split(X,Y,test_size=0.2,random_state=37)
# X_val,X_test,Y_val,Y_test = train_test_split(X_valtest,Y_valtest,test_size=0.5,random_state=37)

print("shape of X_train is:",X_train.shape)
# print("shape of X_Val is:",X_val.shape)
print("shape of X_Test is:", X_test.shape)
print("shape of Y_train is:",Y_train.shape)
# print("shape of Y_Val is:",Y_val.shape)
print("shape of Y_Test is:", Y_test.shape)

num_rows = X_train.shape[1]
num_columns = X_train.shape[2]
num_channels = 1
num_labels = Y_train.shape[1]
X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)   
  


model = Sequential()
model.add(Conv2D(128, kernel_size=3,input_shape=(num_rows, num_columns, num_channels), activation='relu')) 
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=3,input_shape=(num_rows, num_columns, num_channels), activation='relu')) 

model.add(Reshape((X_train.shape[-2], -1)))
model.add(Permute((2, 1)))
          
model.add(Bidirectional(LSTM(128, return_sequences=True)))

model.add(Flatten())

model.add(Dense(num_labels, activation='softmax'))

# opt = Adam(learning_rate=0.0009670460155053197)
# opt = Adam(learning_rate=0.0038766201319368806)

# import visualkeras
# from PIL import ImageFont

# font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
# visualkeras.layered_view(model, legend=True, font=font, scale_xy=0.75, scale_z=0.5).show()

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='CRNN_model_plot.png', show_shapes=True, show_layer_names=True)


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')



score = model.evaluate(X_test, Y_test, verbose=1)
accuracy = 100*score[1]

num_epochs = 37
num_batch_size = 32

history=model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), verbose=1)

model.summary()



# Evaluating the model on the training and testing set
score = model.evaluate(X_train, Y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(X_test, Y_test, verbose=0)
print("Testing Accuracy: ", score[1])

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Predictions
Y_pred = model.predict(X_test, batch_size=32, verbose=1)
np.set_printoptions(precision=5, suppress=True)
print(Y_pred)

mse = (Y_test-Y_pred)**2
print(f"MSE: {mse.mean():0.2f} (+/- {mse.std():0.2f})")

rmse = np.sqrt(mse.mean())
print(f"RMSE: {rmse:0.2f}")

mae = np.abs(Y_test-Y_pred)
print(f"MAE: {mae.mean():0.2f} (+/- {mae.std():0.2f})")

# fpr, tpr, thres = roc_curve(Y_test[:, 1], Y_pred[:, 1])
# roc_auc = auc(fpr, tpr)
# optimal_idx = np.argmax(tpr - fpr)
# optimal_threshold = thres[optimal_idx]
# print("Optimal threshold for class 0:",optimal_threshold)

# tnr=1-fpr
# ba = (tpr+tnr/2)
# print(ba)

    
from sklearn import metrics

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
optimal_idx = dict()
optimal_threshold = dict()
for i in range(num_labels):
        fpr[i], tpr[i], thres = metrics.roc_curve(Y_test[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # print(thres)
        optimal_idx[i] = np.argmax(tpr[i] - fpr[i])
        optimal_threshold[i] = thres[optimal_idx[i]]
        print(f'Threshold value for class{i}:', optimal_threshold[i])
      
        # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], thres = roc_curve(Y_test.ravel(), Y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_labels)]))

      # Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
  
for i in range(num_labels):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
mean_tpr /= num_labels

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
      # Plot all ROC curves
lw=2
plt.figure()
plt.plot(
      fpr["micro"],
      tpr["micro"],
      label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
      color="deeppink",
      linestyle=":",
      linewidth=4,)

plt.plot(
      fpr["macro"],
      tpr["macro"],
      label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
      color="navy",
      linestyle=":",
      linewidth=4,)
# from itertools import cycle
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(num_labels), colors):
      plt.plot(
          fpr[i],
          tpr[i],
          color=color,
          lw=lw,
          label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
      )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC for MLP")
plt.legend(loc="lower right")
plt.show()

  # # Defining the threshold value for comparing each column
# threshold = 0.4
# result= np.where(np.any(Y_pred > threshold, axis=1), np.argmax(Y_pred, axis=1), '6')

# result= result.astype(np.int)
# print(result)
# print(type(result))

# X_test = X_test.reshape(28,3168)
# X_train = X_train.reshape(219, 3168)
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, Y_train)
  
  
# # cross_validate
# cv_scores = cross_val_score(knn, X_train, Y_train, cv=10)

# cv_scores_mean = np.mean(cv_scores)
# print(cv_scores , "\n\n""mean =" ,"{:.2f}".format(cv_scores_mean))
  
# predictions = knn.predict(X_test)
# predictions = np.argmax(predictions, axis=-1)
  
# print(predictions)
# print(predictions.shape)

# print("original Y_test is:", np.argmax(Y_test, axis=-1))

# #compares the result of the thresholding method and the scatterplot
# for i, j in zip(result,predictions):
#      if i == j:
#          result = i
#          print("The predicted class is:", result)
#      else:
#          print("The predicted class is: Mismatch")
