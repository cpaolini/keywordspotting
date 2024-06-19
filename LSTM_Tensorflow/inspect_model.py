import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_inspect

# Load the float model
model = tf.keras.models.load_model('trained_model.h5')

# Define the target DPU configuration
target = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json"

# Define the input shape
input_shape = [None, 99, 32]

# Create the inspector and inspect the model
inspector = vitis_inspect.VitisInspector(target=target)
inspector.inspect_model(
    model,
    input_shape=input_shape,
    dump_model=True,
    dump_model_file="inspect_model.h5",
    plot=True,
    plot_file="model.svg",
    dump_results=True,
    dump_results_file="inspect_results.txt",
    verbose=1
)

