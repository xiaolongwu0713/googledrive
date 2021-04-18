import torch
from torch.utils.data import Dataset, TensorDataset
    
def regulization(net, Lambda):
    w = torch.cat([x.view(-1) for x in net.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err