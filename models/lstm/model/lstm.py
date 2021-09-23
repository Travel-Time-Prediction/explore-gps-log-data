import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_layers * hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        lstm_out, (h_n, _) = self.lstm(x)
        out = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        out = self.dropout(out)
        out = self.linear(out)
        
        return out[:, -1]