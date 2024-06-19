import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# Load the training data
X_train = np.load('X.npy')
print("Loaded X_train shape:", X_train.shape)

# Prepare a subset for calibration
calib_dataset = X_train[:100]
print("Calibration dataset shape:", calib_dataset.shape)

# Ensure the calibration dataset has the correct shape
assert calib_dataset.shape[1:] == (99, 32), "Calibration data shape is incorrect."

# Save the calibration dataset in a suitable format
np.save('calib_dataset.npy', calib_dataset)

# Load the float model
float_model = tf.keras.models.load_model('trained_model.h5')
print("Loaded float model.")

# Load the calibration dataset
calib_dataset = np.load('calib_dataset.npy')
print("Loaded calibration dataset shape:", calib_dataset.shape)

# Perform quantization
quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(
    calib_dataset=calib_dataset,
    calib_steps=100,
    calib_batch_size=10,
    input_shape=[None ,99, 32],
    include_fast_ft=True,
    fast_ft_epochs=10
)

# Save the quantized model
quantized_model.save('quantized_model.h5')
print("Quantization complete. Quantized model saved as 'quantized_model.h5'.")

