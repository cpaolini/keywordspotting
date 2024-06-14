import argparse
import torch
import torch.nn as nn
from pytorch_nndct.apis import Inspector

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

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
parser.add_argument('--target', type=str, required=True, help='Target device')
args = parser.parse_args()

# Load the model
model = LSTMModel(input_len=32, hidden_size=416, num_classes=5, num_layers=2)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 99, 32)

# Create inspector
inspector = Inspector(args.target)

# Inspect the model
inspector.inspect(model, (dummy_input,), device=torch.device('cpu'))

