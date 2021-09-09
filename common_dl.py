import os

from torch import nn
import torch
import numpy as np
import random
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
    if (type(m) == nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

# convert input of (batch, height, width) to (batch, channel, height, width)
class add_channel_dimm(torch.nn.Module):
    def forward(self, x):
        while(len(x.shape) < 4):
            x = x.unsqueeze(1)
        return x


class squeeze_all(torch.nn.Module):
    def forward(self, x):
        return torch.squeeze(x)


def set_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel() # numel: return how many number in that parameter
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def parameterNum(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class myDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        assert self.x.shape[0] == self.y.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.y.shape[0]