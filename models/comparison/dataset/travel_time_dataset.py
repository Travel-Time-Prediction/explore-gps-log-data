import numpy as np
from torch.utils.data import Dataset

class TravelTimeDataset_CNN_LSTM(Dataset):
    def __init__(self, x, y):
        super(TravelTimeDataset_CNN_LSTM, self).__init__()
        self.x = x.reshape((x.shape[0], 1, x.shape[1]))
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])


class TravelTimeDataset_LSTM(Dataset):
    def __init__(self, x, y):
        super(TravelTimeDataset_LSTM, self).__init__()
        x = np.expand_dims(x, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])