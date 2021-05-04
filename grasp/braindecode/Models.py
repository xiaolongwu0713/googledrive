import torch
import torch.nn as nn
import torch.nn.functional as F

# the result is worse, unexpected.
def init_weights(m):
    if (type(m) == nn.Linear or type(m) == nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

class safeLog(nn.Module):
    def __init__(self):
        super(safeLog, self).__init__()
    def forward(self, x):
        return torch.log(torch.clamp(x,min=1e-6))

class squared(nn.Module):
    def __init__(self):
        super(squared, self).__init__()
    def forward(self, x):
        return torch.square(x)
class shallowConv(nn.Module):
    def __init__(self,length,chnNum,convfeature,kernelSize,avgpoolKernel,dropout):  # sampling_rate=1000
        # input_size: EEG channel x datapoint
        super(shallowConv, self).__init__()
        # input data: (batch_size, numFeature, channel/height, time/length)
        testdata = torch.rand((1, 1, chnNum,length))
        self.convt = nn.Conv2d(1, convfeature, kernel_size=(1, kernelSize), stride=1)
        self.convs = nn.Conv2d(convfeature, convfeature, kernel_size=(chnNum, 1), stride=1)
        self.BN = nn.BatchNorm2d(convfeature)
        self.squre = squared()
        self.avgpool = nn.AvgPool2d(kernel_size=(1, avgpoolKernel), stride=(int(avgpoolKernel / 2), 1))
        self.safelog = safeLog()
        self.dropout = nn.Dropout(p=dropout)
        outputshape = self.dropout(self.safelog(self.avgpool(self.squre(self.BN(self.convs(self.convt(testdata)))))))
        self.conv2d = nn.Conv2d(convfeature, 1, kernel_size=(1, outputshape.shape[3]))

    def forward(self, x):
        # torch.Size([1, 28, 1, 90, 1000])
        x=torch.squeeze(x,dim=0)
        output = self.convt(x)
        output = self.convs(output)
        output = self.BN(output)
        output = self.squre(output)
        output = self.avgpool(output)
        output = self.safelog(output)  # (batch_size, plan, channel, time)
        output = self.dropout(output)
        output = self.conv2d(output)
        return torch.squeeze(output)

class deepConv(nn.Module):
    def __init__(self,length,chnNum,convfeature,tkernelSize,blockKernelSize,maxpoolKernel,maxpoolStride,dropout):  # sampling_rate=1000
        # input_size: EEG channel x datapoint
        super(deepConv, self).__init__()
        # input data: (batch_size, numFeature, channel/height, time/length)
        testdata = torch.rand((1, 1, chnNum,length)) #length:1000, chnNum:19
        self.convt = nn.Conv2d(1, convfeature, kernel_size=(1, tkernelSize), stride=1)

        #block 0
        self.convs0 = nn.Conv2d(convfeature, convfeature, kernel_size=(chnNum, 1), stride=1)
        self.BN0 = nn.BatchNorm2d(convfeature)
        self.elu0=nn.ELU()
        self.maxpool0=nn.MaxPool2d((1,maxpoolKernel),(1,maxpoolStride))
        self.dropout0=nn.Dropout(p=dropout)
        output = self.dropout0(self.maxpool0(self.elu0(self.BN0(self.convs0(self.convt(testdata))))))
        #block 1
        self.convs1 = nn.Conv2d(convfeature, 2*convfeature, kernel_size=(1,blockKernelSize), stride=1)
        self.BN1 = nn.BatchNorm2d(2*convfeature)
        self.elu1 = nn.ELU()
        self.maxpool1 = nn.MaxPool2d((1,maxpoolKernel), (1,maxpoolStride))
        self.dropout1 = nn.Dropout(p=dropout)
        #block 2
        self.convs2 = nn.Conv2d(2*convfeature, 4*convfeature, kernel_size=(1,blockKernelSize), stride=1)
        self.BN2 = nn.BatchNorm2d(4*convfeature)
        self.elu2 = nn.ELU()
        self.maxpool2 = nn.MaxPool2d((1,maxpoolKernel), (1,maxpoolStride))
        self.dropout2 = nn.Dropout(p=dropout)
        # block 3
        self.convs3 = nn.Conv2d(4 * convfeature, 8*convfeature, kernel_size=(1,blockKernelSize), stride=1)
        self.BN3 = nn.BatchNorm2d(8*convfeature)
        self.elu3 = nn.ELU()
        self.maxpool3 = nn.MaxPool2d((1,maxpoolKernel), (1,maxpoolStride))

        output = self.dropout1(self.maxpool1(self.elu1(self.BN1(self.convs1(output)))))
        output = self.dropout2(self.maxpool2(self.elu2(self.BN2(self.convs2(output)))))
        output = self.maxpool3(self.elu3(self.BN3(self.convs3(output))))

        self.conv2d = nn.Conv2d(8*convfeature, 1, kernel_size=(1, output.shape[3]))


    def forward(self, x):
        # torch.Size([1, 28, 1, 90, 1000])
        x=torch.squeeze(x,dim=0)
        output = self.convt(x)
        #block 1
        output = self.convs0(output)
        output = self.BN0(output)
        output = self.elu0(output)
        output = self.maxpool0(output)
        output = self.dropout0(output)
        #block 2
        output = self.convs1(output)
        output = self.BN1(output)
        output = self.elu1(output)
        output = self.maxpool1(output)
        output = self.dropout1(output)
        # block 3
        output = self.convs2(output)
        output = self.BN2(output)
        output = self.elu2(output)
        output = self.maxpool2(output)
        output = self.dropout2(output)
        # block 4
        output = self.convs3(output)
        output = self.BN3(output)
        output = self.elu3(output)
        output = self.maxpool3(output)

        output = self.conv2d(output)
        return torch.squeeze(output)