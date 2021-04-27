import torch
from skorch.callbacks import Callback
from skorch.helper import predefined_split
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
import numpy as np
import matplotlib.pyplot as plt

from grasp.TSception.Models import TSception2
from grasp.myskorch import plotPrediction, MyRegressor
from grasp.utils import rawData2,SEEGDataset,set_random_seeds,cuda_or_cup
from grasp.config import *


sid=2
device=cuda_or_cup()
seed = 123456789  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed)


result_dir=root_dir+'grasp/TSception/result_subject2/'
import os
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
sampling_rate=1000
traindata, valdata, testdata = rawData2(sid,'band',activeChannels,move2=True)  # (chns, 15000/15001, 118) (channels, time, trials)
##traindata, valdata, testdata = rawData2('raw','all',move2=True)
traindata = traindata.transpose(2, 0, 1)  # (118, 20, 15000) (trials,channels,  time)
valdata = valdata.transpose(2, 0, 1) # (8, 20, 15000)
testdata = testdata.transpose(2, 0, 1)  # (8, 20, 15000)

#np.save(tmp_dir+'traindata',traindata[:5,:,:])
#np.save(tmp_dir+'valdata',valdata[:5,:,:])
#np.save(tmp_dir+'testdata',testdata[:5,:,:])

#traindata=np.load(tmp_dir+'traindata.npy')
#valdata=np.load(tmp_dir+'valdata.npy')
#testdata=np.load(tmp_dir+'testdata.npy')

trainx, trainy = traindata[:, :-1, :], traindata[:, -1, :] #-2 is real force, -1 is target
valx, valy = valdata[:, :-1, :], valdata[:, -1, :]
testx, testy = testdata[:, :-1, :], testdata[:, -1, :]


learning_rate=0.002
num_T = 3 # (6 conv2d layers) * ( 3 kernel each layer)
num_S = 3
dropout=0.5
Lambda = 1e-6
samples=trainx.shape[0]
chnNum=trainx.shape[1]
step=50 #ms
T=1000 #ms
totalLen=trainx.shape[2] #ms
batch_size=int((totalLen-T)/step) # 280

train_ds = SEEGDataset(trainx, trainy,T, step)
val_ds = SEEGDataset(valx, valy,T, step)
test_ds = SEEGDataset(testx, testy,T, step)

model=TSception2(T, step, sampling_rate,chnNum, num_T, num_S,batch_size,dropout).float()

net = MyRegressor(
    model,
    #train_split=predefined_split(valid_set),
    iterator_train__shuffle=True,
    train_split=predefined_split(val_ds),
    max_epochs=2,
    lr=learning_rate,
    batch_size=1,
    optimizer=torch.optim.Adam,
    criterion = nn.MSELoss,
    callbacks=[('plotPrediction', plotPrediction(result_dir)),],
    device = device
)
net.fit(train_ds,y=None)
