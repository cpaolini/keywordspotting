import os, sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow import keras
print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow_model_optimization.quantization.keras import vitis_quantize
X = np.load('X.npy')
Y = np.load('Y.npy')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

def build_model():
  inputs = tf.keras.Input((193, 87, 1))
  x = tf.keras.layers.Conv2D(32, (7, 7))(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.ReLU(name="relu")(x)

  x = tf.keras.layers.Conv2D(32, (7, 7))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(x)

  x_2 = tf.keras.layers.Conv2D(64, (3, 3))(x)
  x = tf.keras.layers.Conv2D(64, (3, 3))(x)
  x = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(5, activation="sigmoid")(x)

  model = tf.keras.Model(inputs=inputs, outputs=x)
  return model

def main():
  #################################
  ##### build model
  #################################
  model = build_model()
  model.summary()
  #################################
  ##### compile train
  #################################
  model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
  model.fit(X_train, y_train, epochs=2, shuffle=True)
  model.evaluate(X_test, y_test)
  model.save("./float.h5")
  del model

  #################################
  ##### quantize model
  #################################
  loaded_model = tf.keras.models.load_model("./float.h5")
  loaded_model.summary()

  # quantize scope is determined by specify input_layers and output layers
  # ignore layers will not be quantized
  input_layers = ["relu"]
  output_layers = ["flatten"]
  ignore_layers = ["max_pooling2d"]

  quant_model = vitis_quantize.VitisQuantizer(loaded_model, 'pof2s').quantize_model(
		  calib_dataset=X_test,
                  input_layers=input_layers,
                  output_layers=output_layers,
                  ignore_layers=ignore_layers)
  quant_model.summary()
  quant_model.save('quantized.h5')

  with vitis_quantize.quantize_scope():
    quantized_model = tf.keras.models.load_model("quantized.h5")
    quantized_model.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    quantized_model.evaluate(X_test, y_test)


  # Dump Quantized Model
  vitis_quantize.VitisQuantizer.dump_model(quant_model, X_test[0:1],
        "./dump_results", dump_float=True)
if __name__ == '__main__':
  main()
