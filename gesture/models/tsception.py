import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import math
# concate along the plan channel, not the time. Try to test if result is better if reserve physical meaning.
class TSception(nn.Module):
    def __init__(self,chnNum, num_T=64, num_S=64,dropout=0.5):  # sampling_rate=1000
        # input_size: EEG channel x datapoint
        super(TSception, self).__init__()
        # try to use shorter conv kernel to capture high frequency

        # [500, 250, 125, 62, 31, 15, 7]
        # in order to have a same padding, all kernel size shoule be even number, then stride=kernel_size/2
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation

        self.Tception = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, 101), stride=1, padding=(0, 50)),  # kernel: 500
            nn.BatchNorm2d(num_T),
            nn.ReLU())

        self.Sception = nn.Sequential(
            nn.Conv2d(num_S , num_S , kernel_size=(chnNum, 1), stride=1, padding=0),
            nn.BatchNorm2d(num_S),
            nn.ReLU())

        self.lstm = nn.LSTM(272, 128, batch_first=True)

        self.linear=nn.Linear(128,5)
    def forward(self, x):  # ([128, 1, 4, 1024]): (batch_size, )
        self.float()
        #x = torch.randn(1, 208, 500)
        #x = torch.squeeze(x, dim=0)
        x = torch.unsqueeze(x, dim=1) # torch.Size([1, 1, 208, 500])
        y=self.Tception(x)
        y=self.Sception(y) # torch.Size([1, 64, 1, 500])
        y=y.permute(0,2,1,3) # torch.Size([1, 1,64,500])
        y=torch.cat((x,y),2) # torch.Size([1, 1, 272, 500])
        y=torch.squeeze(y,dim=1)
        y = y.permute(0,2,1) # batch_size, time, input_size

        out, _ = self.lstm(y) # torch.Size([1, 272, 128])
        out = self.linear(torch.squeeze(out[:, -1, :]))

        return out
