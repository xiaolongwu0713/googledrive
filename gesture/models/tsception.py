import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import math
# concate along the plan channel, not the time. Try to test if result is better if reserve physical meaning.
class TSception(nn.Module):
    def __init__(self, sampling_rate, chnNum, num_T, num_S,dropout):  # sampling_rate=1000
        # input_size: EEG channel x datapoint
        super(TSception, self).__init__()
        # try to use shorter conv kernel to capture high frequency
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
        win = [int(tt * sampling_rate) for tt in self.inception_window]
        # [500, 250, 125, 62, 31, 15, 7]
        # in order to have a same padding, all kernel size shoule be even number, then stride=kernel_size/2
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation

        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[0]), stride=1, padding=(0, 250)),  # kernel: 500
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[1]), stride=1, padding=(0, 125)),  # 250
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[2] + 1), stride=1, padding=(0, 63)),  # kernel: 126
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[3]), stride=1, padding=(0, 31)),  # kernel:62
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[4] + 1), stride=1, padding=(0, 16)),  # 32
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception6 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[5] + 1), stride=1, padding=(0, 8)),  # 15
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        self.Tception7 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[6] + 1), stride=1, padding=(0, 4)),  # 7
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_S * 6, num_S * 6, kernel_size=(chnNum, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_S * 6, num_S * 6, kernel_size=(int(chnNum * 0.5), 1), stride=(int(chnNum * 0.5), 1),
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception3 = nn.Sequential(
            nn.Conv2d(num_S * 6, num_S * 6, kernel_size=(int(chnNum * 0.5 * 0.5), 1),
                      stride=(int(chnNum * 0.5 * 0.5), 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.BN_t = nn.BatchNorm2d(num_S * 6)
        self.BN_s = nn.BatchNorm2d(num_S * 6)

        self.drop = nn.Dropout(dropout)
        self.avg = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.lstm1 = nn.LSTM(90, 45, batch_first=True)

        self.linear1 = nn.Sequential(
            nn.Linear(45, 5),
            nn.ReLU())
        self.softmax=nn.LogSoftmax(dim=1)
    def forward(self, x):  # ([128, 1, 4, 1024]): (batch_size, )
        self.float()
        x = torch.squeeze(x, dim=0)
        x = torch.unsqueeze(x, dim=1)
        batch_size=x.shape[0]
        # y1 = self.Tception1(x)
        y2 = self.Tception2(x)
        y3 = self.Tception3(x)
        y4 = self.Tception4(x)
        y5 = self.Tception5(x)
        y6 = self.Tception6(x)
        y7 = self.Tception7(x)  # (batch_size, plan, channel, time)
        out = torch.cat((y2, y3, y4, y5, y6, y7), dim=1)  # concate alone plan
        #out = self.BN_t(out) #Todo: braindecode didn't use normalization between t and s filter.

        z1 = self.Sception1(out)
        z2 = self.Sception2(out)
        z3 = self.Sception2(out)
        out_final = torch.cat((z1, z2, z3), dim=2)
        out = self.BN_s(out_final)

        # Todo: test the effect of log(power)
        out = torch.pow(out, 2) # ([28, 18, 5, 15])
        # Braindecode use avgpool2d here
        # out = self.avg(out)
        out = torch.log(out)
        # Todo: test if drop is beneficial
        out = self.drop(out)

        # Todo: try without LSTM.
        out = out.permute(0, 3, 1, 2)  # (batchsize, seq, height, width), ([280, 38, 3, 7])
        seqlen = out.shape[1]
        input_size = int(out.shape[2] * out.shape[3])
        out = out.reshape(batch_size, seqlen, input_size)  # ([280, 38, 21])

        out, _ = self.lstm1(out)
        pred = self.linear1(torch.squeeze(out[:, -1, :]))
        pred = torch.squeeze(pred)
        probs = self.softmax(pred)
        return probs
