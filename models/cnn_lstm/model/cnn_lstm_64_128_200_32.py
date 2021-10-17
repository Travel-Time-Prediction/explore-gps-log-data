import torch
import torch.nn as nn
import torch.nn.functional as F

from model.time_distributed import TimeDistibuted

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = nn.Conv1d(1, 64, kernel_size=2, stride=1)
        self.conv_2 = nn.Conv1d(64, 128, kernel_size=2, stride=1)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        out = F.relu(self.conv_1(x))
        out = F.relu(self.conv_2(out))
        out = self.maxpool_1(out)
        # out = out.view(-1, 128)

        return out

class CNN_LSTM(nn.Module):
    def __init__(self, lstm_hidde_size, lstm_num_layers, dropout=0.5):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN()
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidde_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        self.fc_1 = nn.Linear(lstm_hidde_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        # CNN block
        out = self.cnn(x)
        out = out.view(out.shape[0], out.shape[2], out.shape[1])

        # LSTM block
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        out = self.fc_1(out)
        out = self.dropout(out)
        
        # Output layer
        out = self.output_layer(out)

        return out