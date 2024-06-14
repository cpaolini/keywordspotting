
# LSTM Model Quantization and Inspection

This repository contains scripts for inspecting and quantizing an LSTM model using Vitis AI.

## Prerequisites

- Python 3.8
- PyTorch 1.13.1
- Vitis AI Docker container with the required tools and libraries installed

## Directory Structure

```
workspace/NewModel
├── base.py
├── inspection.py
├── quantization.py
├── X.npy
├── Y.npy
└── trained_lstm_model.pth
```

## Installation

1. **Clone the Repository:**

   ```sh
   git clone <repository-url>
   cd workspace/NewModel
   ```

2. **Install Dependencies:**

   Ensure you have the Vitis AI Docker container running with the required tools and libraries.

## Scripts

### Training the LSTM Model

Run `base.py` to train the LSTM model and save the trained model as `trained_lstm_model.pth`.

```sh
python base.py
```

### Inspecting the Model

Run the `inspection.py` script to inspect the float model.

```sh
python inspection.py --model_path /workspace/LSTM_Pytorch/trained_lstm_model.pth --target DPUCZDX8G_ISA1_B4096
```

### Quantizing the Model

#### Float Model Evaluation

Evaluate the floating-point model.

```sh
python quantization.py --model_path /workspace/LSTM_Pytorch/trained_lstm_model.pth --data_dir /workspace/NewModel --quant_mode float
```

#### Calibration

Run the calibration process to prepare the model for quantization.

```sh
python quantization.py --model_path /workspace/LSTM_Pytorch/trained_lstm_model.pth --data_dir /workspace/NewModel --quant_mode calib
```

#### Test Quantized Model

Test the quantized model.

```sh
python quantization.py --model_path /workspace/LSTM_Pytorch/trained_lstm_model.pth --data_dir /workspace/NewModel --quant_mode test
```

## Files

- `base.py`: Script for training the LSTM model.
- `inspection.py`: Script for inspecting the trained LSTM model.
- `quantization.py`: Script for quantizing the LSTM model.
- `X.npy`: Input data for training.
- `Y.npy`: Labels for training.
- `trained_lstm_model.pth`: Saved trained model.

## License

This project is licensed under the MIT License.
