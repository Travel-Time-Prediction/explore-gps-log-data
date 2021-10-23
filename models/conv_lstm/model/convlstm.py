import torch.nn as nn
import torch

class ConvLSTMBlock(nn.Module):

    def __init__(self, input_channels, hidden_channels, padding, kernel_size):
        super(ConvLSTMBlock, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
       
        self.padding = padding
        self.num_features = 1
        self.conv = nn.Conv1d(in_channels = self.input_channels + self.hidden_channels,
                              out_channels = self.hidden_channels*4, kernel_size = self.kernel_size,padding=padding)                    


    def forward(self, input, h_cur ,c_cur):
        #print(input.shape)
        #print(h_cur.shape)
        combined = torch.cat([input, h_cur], dim=1)  # concatenate along channel axis
        #print(combined.shape)
        #combined = combined.reshape(combined.shape[0],combined.shape[1],combined.shape[2],1)
        #print(combined.shape)

        combined_conv = self.conv(combined)
        #print(combined_conv.shape)
        # (cc_i, cc_f, cc_o, cc_g) = torch.split(combined_conv, int(combined_conv.size()[1] / self.num_features), dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        #print(i.shape)
        f = torch.sigmoid(cc_f)
        #print(f.shape)
        o = torch.sigmoid(cc_o)
       # print(o.shape)
        g = torch.tanh(cc_g)
        #print(g.shape)
        test = f * c_cur
        test2 = i* g
        c_next = (f * c_cur) + (i * g)
        #print(c_next)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvLSTM(nn.Module):
    
    def __init__(self, input_channels,hidden_channels, padding,kernel_size ):
        super(ConvLSTM, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv = ConvLSTMBlock(input_channels,hidden_channels,padding,kernel_size)

    def forward(self, input): #input = frame sequnce (batch_size, num_channels, seq_len, length)
        #print(input.shape)
        input = input.reshape(input.shape[0],input.shape[1],input.shape[2],1)
        batch_size, _, seq_len, length = input.size() #get dimension
        output = torch.zeros(batch_size,self.hidden_channels,seq_len,length,device = device) #Initial output
        H = torch.zeros(batch_size,self.hidden_channels,length,device=device) #Initial hidden state
        C = torch.zeros(batch_size,self.hidden_channels,length,device=device) #Initial cell input
        
        for time_step in range (seq_len):
            H,C = self.conv(input[:,:,time_step],H,C)
            #print(H.shape)
            #print(C.shape)
            output[:,:,time_step] = H
        
        output = output.reshape(output.shape[0],output.shape[1],output.shape[2])

        return output 

class ConvLSTMTravelTime(nn.Module):

    def __init__(self,num_channels,num_kernels,kernel_size,padding,num_layers):
        super(ConvLSTMTravelTime, self).__init__()

        self.sequential = nn.Sequential()
        
        #ADD First Layer
        self.sequential.add_module(
            "convlstm1",ConvLSTM(input_channels=num_channels, hidden_channels=num_kernels,kernel_size=kernel_size,padding=padding)
        )    
        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm1d(num_features=num_kernels) 
        )

        #Add other Layer
        for i in range(2,num_layers+1):
            self.sequential.add_module(
                f"convlstm{i}",ConvLSTM(input_channels=num_kernels, hidden_channels=num_kernels,kernel_size=kernel_size,padding=padding)
            )
            self.sequential.add_module(
                f"batchnorm{i}",nn.BatchNorm1d(num_features=num_kernels)
            )   
        
        # Add Convolutional Layer to predict output 
        #self.conv = nn.Conv1d(in_channels=num_kernels, out_channels=num_channels,kernel_size=kernel_size,padding=1 )

        #add liner Layer to predict output
        self.linear = nn.Linear(in_features=num_kernels,out_features=1)
        
    def forward(self,input):
        output = self.sequential(input)
        output = self.linear(output[:,:,-1])

        return nn.Sigmoid()(output)


       