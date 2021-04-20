import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


################################################## TSception ######################################################
# the result is worse, unexpected.
def init_weights(m):
    if (type(m) == nn.Linear or type(m) == nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


################################################## TSception ######################################################
class TSception(nn.Module):
    def __init__(self, chnNum, sampling_rate, num_T, num_S, batch_size):  # sampling_rate=1000
        # input_size: EEG channel x datapoint
        super(TSception, self).__init__()
        # try to use shorter conv kernel to capture high frequency
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
        win = [int(tt * sampling_rate) for tt in self.inception_window]
        # [500, 250, 125, 62, 31, 15, 7]
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
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[3]), stride=1, padding=(0, 31)),  # kernel:62
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[4] + 1), stride=1, padding=(0, 16)),  # 32
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception6 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[5] + 1), stride=1, padding=(0, 8)),  # 15
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception7 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[6] + 1), stride=1, padding=(0, 4)),  # 7
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(chnNum, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(chnNum * 0.5), 1), stride=(int(chnNum * 0.5), 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception3 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(chnNum * 0.5 * 0.5), 1), stride=(int(chnNum * 0.5 * 0.5), 1),
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)

        self.lstm1 = nn.LSTM(21, 21, batch_first=True)

        self.linear1 = nn.Sequential(
            nn.Linear(21, 1),
            nn.ReLU())

        self.apply(init_weights)

    def forward(self, x):  # ([128, 1, 4, 1024]): (batch_size, )
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception4(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception5(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception6(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)

        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        z = self.Sception3(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)

        # TODO: test the effect of log(power)
        # out = torch.pow(out,2)
        # out = torch.log(out)
        # TODO: test if drop is beneficial
        # out = self.drop(out)

        out = out.permute(0, 3, 1, 2)  # (batchsize, seq, height, width), ([280, 38, 3, 7])
        seqlen = out.shape[1]
        input_size = int(out.shape[2] * out.shape[3])
        out = out.reshape(280, seqlen, input_size)  # ([280, 38, 21])

        out, _ = self.lstm1(out)
        pred = self.linear1(torch.squeeze(out[:, -1, :]))
        return pred


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


################################################## TSception ######################################################
# concate along the plan channel, not the time. Try to test if result is better if reserve physical meaning.
class TSception2(nn.Module):
    def __init__(self, chnNum, sampling_rate, num_T, num_S, batch_size):  # sampling_rate=1000
        # input_size: EEG channel x datapoint
        super(TSception2, self).__init__()
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
            nn.Conv2d(num_T * 6, num_T * 6, kernel_size=(chnNum, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_T * 6, num_T * 6, kernel_size=(int(chnNum * 0.5), 1), stride=(int(chnNum * 0.5), 1),
                      padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        self.Sception3 = nn.Sequential(
            nn.Conv2d(num_T * 6, num_T * 6, kernel_size=(int(chnNum * 0.5 * 0.5), 1),
                      stride=(int(chnNum * 0.5 * 0.5), 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.BN_t = nn.BatchNorm2d(num_T * 6)
        self.BN_s = nn.BatchNorm2d(num_T * 6)

        self.drop = nn.Dropout(0.5)
        self.avg = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.lstm1 = nn.LSTM(90, 45, batch_first=True)

        self.linear1 = nn.Sequential(
            nn.Linear(45, 1),
            nn.ReLU())

    def forward(self, x):  # ([128, 1, 4, 1024]): (batch_size, )
        x = torch.squeeze(x, dim=0)
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
        out = out.reshape(280, seqlen, input_size)  # ([280, 38, 21])

        out, _ = self.lstm1(out)
        pred = self.linear1(torch.squeeze(out[:, -1, :]))
        pred = torch.unsqueeze(pred, dim=0)
        return pred


if __name__ == "__main__":
    model = TSception(2, (4, 1024), 256, 9, 6, 128, 0.2)
    # model = Sception(2,(4,1024),256,6,128,0.2)
    # model = Tception(2,(4,1024),256,9,128,0.2)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
