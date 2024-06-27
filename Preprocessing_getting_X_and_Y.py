# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:29:28 2022

@author: Aarti
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile
import os
import time
import random

import mfccforpreprocessing

import configlib
from configlib import config as C

from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

# from tensorflow.keras.utils import np_utils
from sklearn import metrics
from datetime import datetime


print('started')
start = time.time()


classes = ['binds','can','kite','basic','cv']
#classes = ['blinds','fan','light','music','tv']

def pp():
  data_list=[] #To save paths of all the audio files.....all audio files in list format in data_list
  #data_list-->folder-->files in folder
  for index,label in enumerate(classes):
    class_list=[]
    if label=='silence': #creating silence folder and storing 1sec noise audio files
      silence_path = os.path.join(C["dire"],'silence')
      if not os.path.exists(silence_path):
        os.mkdir(silence_path)
      silence_stride = 2000
      #sample_rate = 16000
      folder = os.path.join(C["dire"],'_background_noise_') #all silence are kept in the background_noise folder

      for file_ in os.listdir(folder):
        if '.wav' in file_:
          load_path = os.path.join(folder,file_)
          sample_rate,y = wavfile.read(load_path)
          for i in range(0,len(y)-sample_rate,silence_stride):
            file_path = "silence/{}_{}.wav".format(file_[:-4],i)
            y_slice = y[i:i+sample_rate]
            wavfile.write(os.path.join(C["dire"],file_path),sample_rate,y_slice)
            class_list.append(file_path)
            
    else:
      folder = os.path.join(C["dire"],label)
      for file_ in os.listdir(folder):
        file_path = '{}/{}'.format(label,file_)    #Ex: up/c9b653a0_nohash_2.wav
        class_list.append(file_path)

    random.shuffle(class_list)              #To shuffle files
    data_list.append(class_list)


  X = []
  Y = []
  preemphasis = 0.985
  print("Feature Extraction Started")
  for i,class_list in enumerate(data_list): #datalist = all files, class list = folder name in datalist, sample = path to the audio file in that particular class list
    for j,samples in enumerate(class_list):    #samples are of the form classes_name/audio file
      if(samples.endswith('.wav')):
        sample_rate,audio = wavfile.read(os.path.join(C["dire"],samples))
        if(audio.size<sample_rate):
            audio = np.pad(audio,(sample_rate-audio.size,0),mode="constant")
        coeff = mfccforpreprocessing.mfcc(audio,sample_rate,preemphasis) # 0.985 = preemphasis
        X.append(coeff)
          #print(X)
        if(samples.split('/')[0] in classes):
            Y.append(samples.split('/')[0])
        elif(samples.split('/')[0]=='_background_noise_'):
            Y.append('silence')
        # print(len(X))
        # print(len(Y))
          
# #X= coefficient array and Y = name of the class
# for fl in mfccwithmulticmd.args['frame_length']:
  print(len(X))
  print(X[0].shape[0])
  print(X[0][0].shape[0])
  A = np.zeros((len(X),X[0].shape[0],X[0][0].shape[0]),dtype='object')
  #print("A shape:",len(X),X[0].shape[0],X[0][0].shape[0])
  #A = np.zeros((len(X),len(X[0]),len(X[0][0])),dtype='float32')
  #print("A shape:",len(X),len(X[0]),len(X[0][0]))
  for i in range(0,len(X)):
    A[i] = np.array(X[i])      #Converting list X into array A
    # print(A.shape)
    
  
  end1 = time.time()
  print("Time taken for feature extraction:{}sec".format(end1-start))

  
  MLB = MultiLabelBinarizer() # one hot encoding for converting labels into binary form
  
  MLB.fit(pd.Series(Y).fillna("missing").str.split(', '))
  Y_MLB = MLB.transform(pd.Series(Y).fillna("missing").str.split(', '))
  MLB.classes_        #Same like classes array
  Y = Y_MLB
  # print("Y is:", Y)
  print("Type of Y is:",type(Y))
  print("Shape of Y:", Y.shape)

  
  X = tf.keras.utils.normalize(X)
  print("Type of X is",type(X))
  print("Shape of X:", X.shape)
  # print("X is:", X)
  
  np.save('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/X_rhyming_word.npy', X)
  # np.save('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/X.npy', X)
  # X_test = np.load('C:/Users/Aarti/.spyder-py3/Checking_with_other_class/X_test.npy')
  
  np.save('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/Y_rhyming_word.npy', Y)
  # np.save('C:/Users/Aarti/.spyder-py3/Use_of_keras_tuner/Y_trial.npy', Y)
  # Y_test = np.load('C:/Users/Aarti/.spyder-py3/Checking_with_other_class/Y_test.npy')
  
  
  
if __name__ == "__main__":
  configlib.parse(save_fname="last_arguments.txt")
  print("Running with configuration:")
  configlib.print_config()
  pp()