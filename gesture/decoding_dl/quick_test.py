import numpy as np
import torch
from gesture.models.deepmodel import deepnet,deepnet_seq,deepnet_rnn, deepnet_da,deepnet_changeDepth,deepnet_expandPlan

net = deepnet_expandPlan(208, 5, 500)
x=np.ones((1,208,500))
out=net(torch.tensor(x).float())



