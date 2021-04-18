import torch
from torch.utils.data import Dataset, TensorDataset

class SEEGDataset(Dataset):
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
    
def regulization(net, Lambda):
    w = torch.cat([x.view(-1) for x in net.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err