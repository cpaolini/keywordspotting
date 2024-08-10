import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile
import os
import time
import random
import librosa
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from keras.callbacks import Callback
import keras.backend as K
import visualkeras


import warnings
warnings.filterwarnings('ignore')

classes = ['blinds', 'fan', 'light', 'music', 'tv']
def apply_preemphasis(audio, preemphasis=0.985):
    return np.append(audio[0], audio[1:] - preemphasis * audio[:-1])


def extract_features():
    data_list = [] 
    
    for label in classes:
        class_list = []
        folder = os.path.join('dataset', label)
        if not os.path.exists(folder):
            print(f"Directory not found: {folder}")
            continue
        for file_ in os.listdir(folder):
            if file_.endswith('.wav'):
                file_path = os.path.join(folder, file_)
                class_list.append(file_path)
        
        if class_list:
            random.shuffle(class_list)  
            data_list.append(class_list)

    if not data_list:
        raise FileNotFoundError("Files not found.")

    X = []
    Y = []
    preemphasis = 0.985

    for i, class_list in enumerate(data_list):
        for samples in class_list:
            try:
                sample_rate, audio = wavfile.read(samples)
                audio = audio.astype(np.float32)
                audio = apply_preemphasis(audio, preemphasis)
            except Exception as e:
                print(f"Error reading {samples}: {e}")
                continue
            if audio.size < sample_rate:
                audio = np.pad(audio, (sample_rate - audio.size, 0), mode="constant")
            

            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
            
            combined = np.vstack((mfcc, chroma, mel, contrast, tonnetz))
            
            X.append(combined)
            Y.append(samples.split(os.path.sep)[-2])  

    if not X or not Y:
        raise ValueError("Files not found.")
    
    print(f"Number of samples: {len(X)}")
    
    # Create a numpy array for X
    X_array_shape = (len(X), X[0].shape[0], X[0].shape[1])
    X_array = np.zeros(X_array_shape, dtype='float32')
    for i in range(len(X)):
        X_array[i] = np.array(X[i])
    
    # One-hot encode the labels
    MLB = MultiLabelBinarizer()
    Y_split = pd.Series(Y).fillna("missing").str.split(', ')
    MLB.fit(Y_split)
    Y_MLB = MLB.transform(Y_split)
    
    print("Shape of Y:", Y_MLB.shape)
    print("Classes:", MLB.classes_)

    # Normalize the features
    X_normalized = tf.keras.utils.normalize(X_array, axis=1)
    
    print("Shape of X:", X_normalized.shape)
    
    return X_normalized, Y_MLB

X, Y = extract_features()

np.save('X.npy', X)
np.save('Y.npy', Y)
import numpy as np
X=np.load("Y.npy")

seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
X_train, X_valtest, Y_train, Y_valtest = train_test_split(X, Y, test_size=0.2, random_state=5)
X_val, X_test, Y_val, Y_test = train_test_split(X_valtest, Y_valtest, test_size=0.5, random_state=5)

print("shape of X_train is:", X_train.shape)
print("shape of X_Test is:", X_test.shape)
print("shape of Y_train is:", Y_train.shape)
print("shape of Y_Test is:", Y_test.shape)

num_rows = X_train.shape[1]
num_columns = X_train.shape[2]
num_channels = 1
num_labels = Y_train.shape[1]

# Reshape data for Conv2D input
X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
X_val = X_val.reshape(X_val.shape[0], num_rows, num_columns, num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)

class LRSchedule(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

lr_schedule = LRSchedule()

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(num_rows, num_columns, num_channels), name="Conv2D_1"))
model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPooling2D_1"))
model.add(BatchNormalization(name="BatchNormalization_1"))
model.add(Dropout(0.3, name="Dropout_1"))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name="Conv2D_2"))
model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPooling2D_2"))
model.add(BatchNormalization(name="BatchNormalization_2"))
model.add(Dropout(0.3, name="Dropout_2"))

model.add(Flatten(name="Flatten"))
model.add(Dense(256, activation='relu', name="Dense_1"))
model.add(BatchNormalization(name="BatchNormalization_3"))
model.add(Dropout(0.5, name="Dropout_3"))
model.add(Dense(num_labels, activation='softmax', name="Output"))

# Visualize the model with labels
visualkeras.layered_view(model, legend=True, to_file='model_visualization_with_labels.png').show()

# Compile the model
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

# Model Training
num_epochs = 200
num_batch_size = 32

history = model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_val, Y_val), callbacks=[lr_schedule], verbose=1)

# Plot accuracy and loss
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Training and Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].legend(loc='upper left')

ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_title('Training and Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(loc='upper left')

plt.tight_layout()
plt.show()

score_train = model.evaluate(X_train, Y_train, verbose=0)
print("Traning Accuracy: ", score_train[1])

score1 = model.evaluate(X_val, Y_val, verbose=0)
print("Validation Accuracy: ", score1[1])

score2 = model.evaluate(X_test, Y_test, verbose=0)
print("Testing Accuracy: ", score2[1])

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
plt.plot(history.history['lr'])
plt.title('Learning Rate Schedule')
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')
plt.show()
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Plot training & validation accuracy values
ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Training and Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')  # Set x-axis label
ax[0].legend(loc='upper left')

# Plot training & validation loss values
ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_title('Training and Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')  # Set x-axis label
ax[1].legend(loc='upper left')

plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Predict the probabilities from the validation dataset
Y_pred_proba = model.predict(X_val)

# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis=1)

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(Y_val.ravel(), Y_pred_proba.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

# Predict the probabilities from the validation dataset
Y_pred_proba = model.predict(X_val)

# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis=1)

# Compute precision and recall for each class
precision, recall, _ = precision_recall_curve(Y_val.ravel(), Y_pred_proba.ravel())

# Plot the precision-recall curve
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Make predictions on the test set
Y_pred_prob = model.predict(X_test)
Y_pred = np.argmax(Y_pred_prob, axis=1)
Y_test_labels = np.argmax(Y_test, axis=1)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(Y_test_labels, Y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(Y_test_labels, Y_pred, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Alternatively, use classification_report to get a detailed report
report = classification_report(Y_test_labels, Y_pred, target_names=classes)
print("Classification Report:")
print(report)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(num_rows, num_columns, num_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_labels, activation='softmax'))

# Compile the model
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
# Model Training
num_epochs = 200
num_batch_size = 32
model.fit(X, Y, batch_size=num_batch_size, epochs=num_epochs, verbose=False)
# Save the Keras model to a .h5 file
model_save_path = 'model.h5'
model.save(model_save_path)
print(f"Keras model saved to {model_save_path}")
print('model trained')