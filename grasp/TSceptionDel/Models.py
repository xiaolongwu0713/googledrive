import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

################################################## TSception ######################################################
class TSception(nn.Module):
    def __init__(self,chnNum, sampling_rate, num_T, num_S, batch_size):
        # input_size: EEG channel x datapoint
        self.batch_size=batch_size
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[0]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[1]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[2]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception4 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[3] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception5 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[4] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))
        self.Tception6 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[5] * sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)))

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(chnNum,1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8)))
        self.Sception2 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(chnNum*0.5),1), stride=(int(chnNum*0.5),1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8)))
        self.Sception3 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(int(chnNum * 0.5 * 0.5), 1), stride=(int(chnNum * 0.5 * 0.5), 1),padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)))

        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)

        self.lstm1 = nn.LSTM(21, 21, batch_first=True)

        self.linear1 = nn.Sequential(
            nn.Linear(21, 1),
            nn.ReLU())
        
    def forward(self, x): # ([128, 1, 4, 1024]): (batch_size, )
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out,y),dim = -1)
        y = self.Tception3(x)
        out = torch.cat((out,y),dim = -1)
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
        out_final = torch.cat((out_final,z),dim = 2)
        z = self.Sception3(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)

        out = out.permute(0, 3, 1, 2) #(batchsize, seq, height, width), ([280, 38, 3, 7])
        seqlen=out.shape[1]
        input_size=int(out.shape[2] * out.shape[3])
        out = out.reshape(self.batch_size, seqlen, input_size) # ([280, 38, 21])

        out, _ = self.lstm1(out)
        pred = self.linear1(torch.squeeze(out[:,-1,:]))
        return pred

if __name__ == "__main__":
    model = TSception(2,(4,1024),256,9,6,128,0.2)
    #model = Sception(2,(4,1024),256,6,128,0.2)
    #model = Tception(2,(4,1024),256,9,128,0.2)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
