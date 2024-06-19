import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load the data
X = np.load('X.npy')
Y = np.load('Y.npy')

# Split the data
X_train, X_valtest, Y_train, Y_valtest = train_test_split(X, Y, test_size=0.2, random_state=37)
X_val, X_test, Y_val, Y_test = train_test_split(X_valtest, Y_valtest, test_size=0.5, random_state=37)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of Y_train:", Y_train.shape)
print("Shape of Y_test:", Y_test.shape)

# Define the model
num_rows = X_train.shape[1]
num_columns = X_train.shape[2]
num_labels = Y_train.shape[1]

model = Sequential()
model.add(LSTM(416, return_sequences=True, input_shape=(num_rows, num_columns)))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(num_labels, activation='softmax'))

# Compile the model
opt = Adam(learning_rate=0.0008711452772441899)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

# Train the model
num_epochs = 29
num_batch_size = 32

model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), verbose=1)

model.summary()

score = model.evaluate(X_train, Y_train, verbose=0)
print("Training Accuracy: ", score[1])

score1 = model.evaluate(X_test, Y_test, verbose=0)
print("Testing Accuracy: ", score1[1])

# Save the model
model.save('trained_model.h5')
