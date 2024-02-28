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

# X = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/X.npy')
# Y = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/Y.npy')

X_train = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/X.npy')
Y_train = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/Y.npy')

# X_test = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/X_rhyming_word.npy')
# Y_test = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/Y_rhyming_word.npy')

X_test = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/X_test_5_files_each_class.npy')
Y_test = np.load('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/Y_test_5_files_each_class.npy')

  
# X_train,X_valtest,Y_train,Y_valtest = train_test_split(X,Y,test_size=0.2,random_state=37)
# X_val,X_test,Y_val,Y_test = train_test_split(X_valtest,Y_valtest,test_size=0.5,random_state=37)

print("shape of X_train is:",X_train.shape)
# print("shape of X_Val is:",X_val.shape)
print("shape of X_Test is:", X_test.shape)
print("shape of Y_train is:",Y_train.shape)
# print("shape of Y_Val is:",Y_val.shape)
print("shape of Y_Test is:", Y_test.shape)


# def build_model2(hp):
#       model = tf.keras.Sequential()
#       # model.add(Flatten())
#       for i in range(hp.Int('layers', 1, 3)):
#           model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 32, 512, step=32), 
#                                           activation = hp.Choice('act_' + str(i), ['relu', 'sigmoid','tanh'])))
#       model.add(Flatten())
#       model.add(layers.Dense(5, activation='softmax'))
#       learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
#       model.compile(keras.optimizers.Adam(learning_rate=learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
#       return model

num_labels = Y_train.shape[1]

model = Sequential()

model.add(Dense(480, activation='relu'))
# model.add(Dense(96, activation='sigmoid'))
# model.add(Dense(192, activation='sigmoid'))

model.add(Flatten())


model.add(Dense(num_labels, activation='softmax'))

opt = Adam(learning_rate=0.0009670460155053197)
# opt = Adam(learning_rate=0.0038766201319368806)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)


score = model.evaluate(X_test, Y_test, verbose=1)
accuracy = 100*score[1]

num_epochs = 10
num_batch_size = 32

history=model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), verbose=1)

model.summary()

##for visualization of the model, use the following code
# import visualkeras
# from PIL import ImageFont

# font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
# visualkeras.layered_view(model, legend=True, font=font, scale_xy=1, scale_z=0.5, max_z=100).show()

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='MLP_model_plot.png', show_shapes=True, show_layer_names=True)
##---------


# Evaluating the model on the training and testing set
score = model.evaluate(X_train, Y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(X_test, Y_test, verbose=0)
print("Testing Accuracy: ", score[1])

# summarize history for accuracy
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 12

# plt.rcParams.update({'font.size': 10})

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
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


# thresh_vals = np.array([0.5643991, 0.13516375, 0.42182505, 0.89806145, 0.121829145])
# m_thresh = np.repeat(thresh_vals.reshape(1,5), 28, axis=0)

# rep_vals = np.array(['0', '0', '0', '0', '0'])
# m_rep = np.repeat(rep_vals.reshape(1,5), 28, axis=0)

# mask = Y_pred < thresh_vals
# Y_pred[mask] = m_rep[mask]
# print('After thresholding:', Y_pred)

# result = np.where(np.amax(Y_pred,axis=-1), np.argmax(Y_pred, axis=-1), '6')
# result= result.astype(np.int64)
# print("Thresholding result:",result)

# ###for knn
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
  
# print("Knn result:",predictions)
# # print(predictions.shape)
# aaa=np.argmax(Y_test,axis=-1)
# print("original Y_test is:", aaa)


# # #compares the result of the thresholding method and the scatterplot
# # for i, j,k in zip(result,predictions, aaa):
# #       if i == j and j ==k:
# #           result = k
# #           print("The predicted class is:", result)
# #       else:
# #           print("The predicted class is: Mismatch")

# # #For printing confusion matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(aaa,result)
# cm1 = confusion_matrix(aaa,predictions)
# print(cm)
# print(cm1)

# ###for printing the confusion matrix
# # fig, ax = plt.subplots(figsize=(5, 5))
# # ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
# # for i in range(cm.shape[0]):
# #     for j in range(cm.shape[1]):
# #         ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
# # plt.xlabel('Predictions', fontsize=18)
# # plt.ylabel('Actuals', fontsize=18)
# # plt.title('Confusion Matrix', fontsize=18)
# # plt.show()

# from sklearn.metrics import classification_report,accuracy_score,precision_recall_fscore_support
# ##for classification report containing precision, f1 score and recall for each class
# # print(classification_report(aaa,result))
# # print(classification_report(aaa,predictions))

# ##for classification accuracy

# print("Accuracy for thresholding method:",accuracy_score(aaa,result)*100)
# print("Accuracy for knn method:",accuracy_score(aaa,predictions)*100)

# ###for finding precision, recall and fscore
# print("Report for thresholding:",precision_recall_fscore_support(aaa,result, average='macro'))
# print("Report for knn:",precision_recall_fscore_support(aaa,predictions, average='macro'))

# # from pycm import *
# # cm = ConfusionMatrix(actual_vector=aaa, predict_vector=result)
# # cm.print_matrix()
# # cm.stat(summary=True)