import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import math

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

def init_weights(m):
    if (type(m) == nn.Linear or type(m) == nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

class SelectionLayer(nn.Module):
    def __init__(self, test_shape,N, M, temperature=1.0):

        super(SelectionLayer, self).__init__()
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.N = N
        self.M = M
        self.qz_loga = Parameter(torch.randn(N, M) / 100)  # ~N(0,1)

        self.temperature = self.floatTensor([temperature])
        self.freeze = False
        self.thresh = 8.0
        test_input=torch.ones(test_shape)
        if torch.cuda.is_available():
            test_input.cuda()
            self.cuda()
        self.out=self.forward(test_input)

    def quantile_concrete(self, x):  # eq: 2

        if torch.cuda.is_available():
            x.cuda()
            self.cuda()
            self.temperature.cuda()
            self.qz_loga.cuda()
        g = -torch.log(-torch.log(x))  # gumbel distribution

        y = (self.qz_loga + g) / self.temperature  # beta
        y = torch.softmax(y, dim=1)  # concrete distribution. torch.Size([16, 44, 3])

        return y

    def regularization(self):

        eps = 1e-10
        z = torch.clamp(torch.softmax(self.qz_loga, dim=0), eps, 1)
        H = torch.sum(F.relu(torch.norm(z, 1, dim=1) - self.thresh)) # calculate the 1-norm: sum of absolute value

        return H

    def get_eps(self, size):

        eps = self.floatTensor(size).uniform_(epsilon, 1 - epsilon)

        return eps

    def sample_z(self, batch_size, training):

        if training:
            eps = self.get_eps(self.floatTensor(batch_size, self.N, self.M))
            z = self.quantile_concrete(eps)  # eq: 2  torch.Size([16, 44, 3])
            z = z.view(z.size(0), 1, z.size(1), z.size(2))  # torch.Size([16, 1, 44, 3])
            return z
        else:
            ind = torch.argmax(self.qz_loga, dim=0)
            one_hot = self.floatTensor(np.zeros((self.N, self.M)))
            for j in range(self.M):
                one_hot[ind[j], j] = 1
            one_hot = one_hot.view(1, 1, one_hot.size(0), one_hot.size(1))
            one_hot = one_hot.expand(batch_size, 1, one_hot.size(2), one_hot.size(3))
            return one_hot

    def forward(self, x):
        z = self.sample_z(x.size(0), training=(self.training and not self.freeze))  # torch.Size([16, 1, 44, 3])
        z_t = torch.transpose(z, 2, 3)  # torch.Size([16, 1, 3, 44])
        if torch.cuda.is_available():
            x.cuda()
            z_t.cuda()
        if not x.is_cuda:
            return
        out = torch.matmul(z_t, x)  # x:torch.Size([16, 1, 44, 1125])
        return out  # out: torch.Size([16, 1, 3, 1125])

# concate along the plan channel, not the time. Try to test if result is better if reserve physical meaning.
class TSception2(nn.Module):
    def __init__(self, sampling_rate, chnNum, num_T, num_S,dropout):  # sampling_rate=1000
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
            nn.Linear(45, 1),
            nn.ReLU())

    def forward(self, x):  # ([128, 1, 4, 1024]): (batch_size, )
        self.float()
        x = torch.squeeze(x, dim=0)
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
        pred = torch.unsqueeze(pred, dim=0)
        return pred


# concate along the plan channel, not the time. Try to test if result is better if reserve physical meaning.
class TSception_small(nn.Module):
    def __init__(self, test_shape, sampling_rate, chnNum, num_T, num_S,dropout):  # sampling_rate=1000
        # input_size: EEG channel x datapoint
        super(TSception_small, self).__init__()
        input=torch.ones(test_shape)
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
        out1=self.Tception1(input)
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[1]), stride=1, padding=(0, 125)),  # 250
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        out2 = self.Tception2(input)
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[2] + 1), stride=1, padding=(0, 63)),  # kernel: 126
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        out3 = self.Tception3(input)
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, win[3]), stride=1, padding=(0, 31)),  # kernel:62
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)))
        out4 = self.Tception4(input)
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
        out_t = torch.cat((out1, out2), dim=1)  # concate alone plan

        plan_num=out_t.shape[1]
        self.Sception1 = nn.Sequential(
            nn.Conv2d(plan_num, plan_num, kernel_size=(chnNum, 1), stride=1, padding=0),
            nn.ReLU())
            #nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        outs1 = self.Sception1(out_t)
        self.Sception2 = nn.Sequential(
            nn.Conv2d(plan_num, plan_num, kernel_size=(int(chnNum * 0.5), 1), stride=(int(chnNum * 0.5), 1),
                      padding=0),
            nn.ReLU())
            #nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        outs2 = self.Sception2(out_t)
        self.Sception3 = nn.Sequential(
            nn.Conv2d(plan_num, plan_num, kernel_size=(int(chnNum * 0.5 * 0.5), 1),
                      stride=(int(chnNum * 0.5 * 0.5), 1), padding=0),
            nn.ReLU())
            #nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))
        outs3 = self.Sception3(out_t)
        out_s = torch.cat((outs1, outs2), dim=2)


        #self.BN_t = nn.BatchNorm2d(num_S * 6)
        self.BN_s = nn.BatchNorm2d(out_s.shape[1])

        self.drop = nn.Dropout(dropout)
        self.avg = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5))
        out=self.avg(out_s)

        out = out.permute(0, 3, 1, 2)  # (batchsize, seq, height, width), ([280, 38, 3, 7])
        seqlen = out.shape[1]
        input_size = int(out.shape[2] * out.shape[3])
        out = out.reshape(out.shape[0], seqlen, input_size)  # ([280, 38, 21])
        input_feature=out.shape[2]

        self.lstm1 = nn.LSTM(input_feature, int(input_feature/2), batch_first=True)
        out, _=self.lstm1(out)
        self.linear1 = nn.Sequential(
            nn.Linear(int(input_feature/2), 1),
            nn.ReLU())

    def forward(self, x):  # ([128, 1, 4, 1024]): (batch_size, )
        self.float()
        x = torch.squeeze(x, dim=0)
        batch_size=x.shape[0]
        y1 = self.Tception1(x)
        y2 = self.Tception2(x)
        #y3 = self.Tception3(x)
        #y4 = self.Tception4(x)
        #y5 = self.Tception5(x)
        #y6 = self.Tception6(x)
        #y7 = self.Tception7(x)  # (batch_size, plan, channel, time)
        out = torch.cat((y1, y2), dim=1)  # concate alone plan
        #out = self.BN_t(out) #Todo: braindecode didn't use normalization between t and s filter.

        z1 = self.Sception1(out)
        z2 = self.Sception2(out)
        #z3 = self.Sception2(out)
        out_final = torch.cat((z1, z2), dim=2)
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
        pred = torch.unsqueeze(pred, dim=0)
        return pred

class wholenet(nn.Module):
    def __init__(self, test_shape, enable_select,input_dim, M ,sampling_rate, chnNum, num_T, num_S,dropout):
        super(wholenet, self).__init__()
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.M = M
        self.N = input_dim

        self.enable_select = True
        if enable_select:
            self.selection_layer = SelectionLayer(test_shape,self.N, self.M)
            self.select_output=self.selection_layer.out
            test_shape=self.select_output.shape

        self.network = TSception_small(test_shape,sampling_rate, chnNum, num_T, num_S, dropout)
        self.layers = self.create_layers_field()
        self.apply(init_weights)

    def create_layers_field(self):
        layers = []
        for idx, m in enumerate(self.modules()):
            if (type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == SelectionLayer):
                layers.append(m)
        return layers

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        if self.enable_select==True:
            x = self.selection_layer(x)
        out = self.network(x)
        return out

    def monitor(self):
        m = self.selection_layer
        eps = 1e-10
        # Probability distributions
        z = torch.clamp(torch.softmax(m.qz_loga, dim=0), eps, 1)
        # Normalized entropy
        H = - torch.sum(z * torch.log(z), dim=0) / math.log(self.N)
        # Selections
        s = torch.argmax(m.qz_loga, dim=0) + 1

        return H, s, z

    def set_temperature(self, temp):
        m = self.selection_layer
        m.temperature = temp

    def set_thresh(self, thresh):
        m = self.selection_layer
        m.thresh = thresh

    def set_freeze(self, x):
        m = self.selection_layer
        if (x):
            for param in m.parameters():
                param.requires_grad = False
            m.freeze = True
        else:
            for param in m.parameters():
                param.requires_grad = True
            m.freeze = False

    def regularizer(self, lamba, weight_decay):
        # Regularization of selection layer
        reg_selection = self.floatTensor([0])
        # L2-Regularization of other layers
        reg = self.floatTensor([0])
        for i, layer in enumerate(self.layers):
            if (type(layer) == SelectionLayer):
                if (self.enable_select==True):
                    reg_selection += layer.regularization() # 1-norm regularization
            else:
                #reg += torch.sum(torch.pow(layer.weight, 2)) # 2-norm regularization
                reg += torch.sum(torch.abs(layer.weight)) # abs sum
        reg = weight_decay * reg + lamba * reg_selection
        return reg

