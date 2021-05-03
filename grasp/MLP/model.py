import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch
# model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, 80)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = nn.Sigmoid()
        self.hidden2 = nn.Linear(80, 40)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = nn.Sigmoid()
        self.hidden3 = nn.Linear(40, 10)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Sigmoid()
        self.hidden4 = nn.Linear(10, 8)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = nn.Sigmoid()
        self.lastlayer = nn.Linear(8, 1)
        xavier_uniform_(self.lastlayer.weight)


    # forward propagate input
    def forward(self, X):
        X = self.hidden1(torch.squeeze(X).T)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.hidden4(X)
        X = self.act4(X)
        X = self.lastlayer(X)
        return X
