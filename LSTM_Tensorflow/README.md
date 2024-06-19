
# LSTM Model Training, Inspection, and Quantization with Vitis AI 3.5

This repository contains scripts to train an LSTM model, inspect the trained model, and perform quantization using Vitis AI 3.5 on a ZCU104 board.

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Scikit-learn
- Vitis AI 3.5 Docker Image for TensorFlow 2.x
- Vitis AI TensorFlow Model Optimization Toolkit

## Setup

1. Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Ensure you have Docker installed and configured. Pull the Vitis AI Docker image:
    ```bash
    docker pull xilinx/vitis-ai-tensorflow2:latest
    ```

3. Start the Docker container:
    ```bash
    docker run -it --rm -v $(pwd):/workspace xilinx/vitis-ai-tensorflow2:latest
    ```

## Training the Model

1. Place your training data (`X.npy` and `Y.npy`) in the repository directory.

2. Run the training script:
    ```bash
    python train_save_model.py
    ```

This script will train the LSTM model and save it as `trained_model.h5`.

## Inspecting the Model

1. Run the model inspection script:
    ```bash
    python inspect_model.py
    ```

This script will inspect the trained model and save the inspected model as `inspect_model.h5`. It will also generate a plot (`model.svg`) and dump the inspection results to `inspect_results.txt`.

## Quantizing the Model

1. Run the quantization script:
    ```bash
    python quantize_model.py
    ```

This script will perform quantization on the trained model using a subset of the training data for calibration. The quantized model will be saved as `quantized_model.h5`.

## Troubleshooting

- Ensure your data files (`X.npy` and `Y.npy`) are correctly formatted and located in the repository directory.
- If you encounter any errors related to TensorFlow or Vitis AI, make sure you are using the correct Docker image and have installed all required dependencies.

## Additional Notes

- Modify the training script (`train_save_model.py`) if you need to change model parameters, training epochs, or batch size.
- Review the inspection and quantization scripts (`inspect_model.py` and `quantize_model.py`) for additional configurations and parameters.

## References

- [Vitis AI Documentation](https://xilinx.github.io/Vitis-AI/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
