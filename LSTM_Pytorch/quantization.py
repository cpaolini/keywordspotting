import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_len, hidden_size, num_classes, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.6)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(hidden_size * 99, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.flatten(out)
        out = self.dense(out)
        out = self.softmax(out)
        return out

def load_data(data_dir):
    # Load the data
    X = np.load(f'{data_dir}/X.npy')
    Y = np.load(f'{data_dir}/Y.npy')

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # Split the data
    X_train, X_valtest, Y_train, Y_valtest = train_test_split(X, Y, test_size=0.2, random_state=37)
    X_val, X_test, Y_val, Y_test = train_test_split(X_valtest, Y_valtest, test_size=0.5, random_state=37)
    
    return X_train, X_test, Y_train, Y_test

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()
    return 100 * correct / total

def main(args):
    # Load data
    X_train, X_test, Y_train, Y_test = load_data(args.data_dir)
    
    sequence_len = 99
    input_len = 32
    hidden_size = 416
    num_classes = 5
    num_layers = 2

    # Initialize the model
    model = LSTMModel(input_len=input_len, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    if args.quant_mode == 'float':
        # Evaluate float model
        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        accuracy = evaluate(model, test_loader)
        print(f'Float Model Testing Accuracy: {accuracy}%')
    
    elif args.quant_mode == 'calib':
        # Perform calibration
        dummy_input = torch.randn(1, sequence_len, input_len)
        quantizer = torch_quantizer('calib', model, (dummy_input,), device=torch.device('cpu'))
        quant_model = quantizer.quant_model

        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        with torch.no_grad():
            for inputs, _ in train_loader:
                quant_model(inputs)
                break  # Only run one batch for calibration

        quantizer.export_quant_config()
        quantizer.export_xmodel(output_dir="quantize_result", deploy_check=True)
        print("Quantization completed. Check the 'quantize_result' directory for the results.")

    elif args.quant_mode == 'test':
        # Load quantized model and test
        dummy_input = torch.randn(1, sequence_len, input_len)
        quantizer = torch_quantizer('test', model, (dummy_input,), device=torch.device('cpu'))
        quant_model = quantizer.quant_model

        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        quant_accuracy = evaluate(quant_model, test_loader)
        print(f'Testing Accuracy of Quantized Model: {quant_accuracy}%')

        # Export the xmodel
        quantizer.export_xmodel(output_dir="quantize_result", deploy_check=True)
        print("Xmodel exported. Check the 'quantize_result' directory for the xmodel file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--quant_mode', type=str, required=True, choices=['float', 'calib', 'test'], help='Quantization mode')
    args = parser.parse_args()
    main(args)

