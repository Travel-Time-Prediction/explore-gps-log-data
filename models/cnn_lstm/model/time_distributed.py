import torch.nn as nn
from torch.nn.modules.container import ModuleList

class TimeDistibuted(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistibuted, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, 1, x.size(-1))

        print(x_reshape.shape)
        print(type(x_reshape))
        
        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1)) # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1)) # (timesteps, samples, output_size)

        return y