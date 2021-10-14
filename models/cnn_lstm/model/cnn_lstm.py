import torch
import torch.nn as nn
from torch.nn.modules import batchnorm

from time_distributed import TimeDistibuted

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = TimeDistibuted(nn.Conv1d(1, 64, kernel_size=2))
        self.conv_2 = TimeDistibuted(nn.Conv1d(64, 128, kernel_size=2))
        self.maxpool_1 = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.maxpool_1(out)

        return out

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN()
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        out = self.cnn(x)
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)