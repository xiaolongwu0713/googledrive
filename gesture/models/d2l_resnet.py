import torch
from torch import nn
from torch.nn import functional as F
#from d2l import torch as d2l

from common_dl import add_channel_dimm, squeeze_all


class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        # keep width and hight by using kernel_size=3 and pedding=1;
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True,
                         strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class d2lresnet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.block_num=block_num
        b1 = nn.Sequential(add_channel_dimm(),nn.Conv2d(1, 64, kernel_size=(1,50), stride=(1,1)),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=(3,3), stride=(1,1)))
        b2 = nn.Sequential(*resnet_block(64, 64, 1, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 1))
        b4 = nn.Sequential(*resnet_block(128, 256, 1))
        b5 = nn.Sequential(*resnet_block(256, 512, 1))
        b6 = nn.Sequential(*resnet_block(512, 1024, 1))

        self.d2lresnet = nn.Sequential(b1, b2, b3, b4, b5,b6, nn.AdaptiveAvgPool2d((1, 1)),squeeze_all(),nn.Linear(1024, 5))

    def forward(self,x):
        return self.d2lresnet(x) # use CrossEntropyLoss loss

#x=torch.randn(32,1,208,500)






