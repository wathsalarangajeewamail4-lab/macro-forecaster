import torch
import torch.nn as nn
import numpy as np

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_assets):
        super(MultiTaskLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Shared Macro Encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Asset-specific heads
        # Each head predicts the return for a specific asset
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_assets)
        ])
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # We only need the output from the last time step
        out, _ = self.lstm(x, (h0, c0))
        last_out = out[:, -1, :] # Shape: (batch, hidden_dim)
        
        # Pass through each asset-specific head
        predictions = [head(last_out) for head in self.heads]
        
        # Concatenate along the feature dimension
        return torch.cat(predictions, dim=1) # Shape: (batch, num_assets)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    # Assume data is a numpy array where first columns are the targets
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length, :4] # Target is next day's returns for the 4 assets
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
