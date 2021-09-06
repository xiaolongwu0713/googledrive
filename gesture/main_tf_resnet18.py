#%cd /content/drive/MyDrive/
# raw_data is imported from global config

#%%capture
#! pip install mne==0.19.2;
#! pip install torch==1.7.0;
#! pip install hdf5storage;
#! pip install skorch==0.10.0;
#! pip install Braindecode==0.5.1;

import hdf5storage
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from gesture.models.resnet import _my_resnet18, my_resnet18
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier

from gesture.config import *
from common_dl import myDataset, set_random_seeds
from myskorch import on_epoch_begin_callback, on_batch_end_callback


sid = 10  # 4

data_dir = data_dir + 'preprocessing/P' + str(sid) + '/tfInput/'
filename=data_dir+'dataset.hdf5'
f1 = h5py.File(filename, "r")
list(f1.keys())
X_train = f1['X_train'][:] # (650, 10, 148, 250)
X_test = f1['X_test'][:] #(50, 10, 148, 250)
y_train = f1['y_train'][:] # (650,)
y_test = f1['y_test'][:]
f1.close()

train_ds = myDataset(X_train, y_train)
val_ds = myDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_ds, batch_size=5, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_ds, batch_size=5, pin_memory=False)
#test it
#(x,y)=iter(train_loader).next() # x: torch.Size([1, 10, 148, 250]); y: torch.Size([1])


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed) # same as braindecode random seeding

_net=_my_resnet18(10,5, pretrained=True).float()
net=my_resnet18(_net)
if cuda:
    net.cuda()
#x= torch.randn(1, 10, 148, 250)
#net(x).shape

lr = 0.0001
weight_decay = 1e-10
batch_size = 1
epoch_num = 100
criterion=torch.nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(epoch_num):
    print("------ epoch " + str(epoch) + " -----")
    net.train()

    loss_epoch = 0
    # trial=0
    for _, (trainx, trainy) in enumerate(train_loader):  # ([1, 15000, 19]), ([1, 15000])
        optimizer.zero_grad()
        if (cuda):
            trainx = trainx.float().cuda()
            trainy = trainy.float().cuda()
        else:
            #pass
            trainx = trainx.float()
            #target = trainy.float()
        y_pred = net(trainx)
        # target = torch.from_numpy(target)

        # regularization
        loss = criterion(y_pred, trainy.type(torch.LongTensor))
        # loss3 = y_pred.cpu().detach().numpy()
        # loss3 = np.std(np.diff(loss3.reshape(-1)))
        loss.backward()
        optimizer.step()

        ls = loss.item()
        loss_epoch += ls
    mean_loss=loss_epoch/len(train_loader)
    print("Epoch " + str(epoch) + " loss:" + str(mean_loss) + ".")
    if epoch % 2 == 0:
        net.eval()
        print("Validating...")
        with torch.no_grad():
            target=[]
            predict=[]
            for _, (val_x, val_y) in enumerate(val_loader):  # ([1, 15000, 19]), ([1, 15000])
                if (cuda):
                    val_x = val_x.float().cuda()
                    val_y = val_y.float().cuda()
                else:
                    val_x = val_x.float()
                    #val_y = val_y.float()
                y_pred = net(val_x)
                tmp=[a.tolist().index(max(a.tolist())) for a in y_pred.numpy()]
                predict=predict+tmp
                target=target+val_y
            accuracy=accuracy_score(target, predict)
        print("Accuracy: " + str(accuracy) + ".")

