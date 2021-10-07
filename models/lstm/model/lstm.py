import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM_Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(num_layers * hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batch_size = x.shape[0]

        x = F.relu(self.linear1(x))

        lstm_out, (h_n, _) = self.lstm(x)
        out = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        out = self.dropout(out)
        out = self.linear2(out)
        
        return out[:, -1]