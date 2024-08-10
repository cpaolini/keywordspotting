import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
X = np.load('X.npy')
Y = np.load('Y.npy')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

def D_Model():
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
    model = D_Model()
    model.summary()
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=2, shuffle=True)
    model.evaluate(X_test, y_test)
    model.save("./model.h5")
    del model

    loaded_model = tf.keras.models.load_model("./model.h5")
    loaded_model.summary()

    input_layers = ["relu"]
    output_layers = ["flatten"]
    ignore_layers = ["max_pooling2d"]

    quant_model = vitis_quantize.VitisQuantizer(loaded_model, 'pof2s').quantize_model(
        calib_dataset=X_test,
        input_layers=input_layers,
        output_layers=output_layers,
        ignore_layers=ignore_layers)
    quant_model.save('quantized_model.h5')

    print("======================================================================")
    print("======================================================================")
    print("Quantized model performance")
    with vitis_quantize.quantize_scope():
        quantized_model = tf.keras.models.load_model("quantized_model.h5")
        quantized_model.compile(optimizer="adam",
                                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        quantized_model.evaluate(X_test, y_test)
    print("======================================================================")
    print("======================================================================")
    vitis_quantize.VitisQuantizer.dump_model(quant_model, X_test[0:1],
                                             "./dump_results", dump_float=True)

    quantized_model.save('saved_model_quantized')

    full_model = tf.function(lambda x: quantized_model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(quantized_model.inputs[0].shape, quantized_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="frozen_model.pb",
                      as_text=False)

if __name__ == '__main__':
    main()
