import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from grasp.TSception.utils import regulization
from grasp.utils import freq_input, SEEGDataset3D, cuda_or_cup, set_random_seeds, raw_input
from grasp.braindecode.Models import shallowConv,deepConv
from examples.IMV_LSTM.networks import IMVTensorLSTM
# load the data: regression to target force derivative
from grasp.config import *

device=cuda_or_cup()
seed = 123456789  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed)

result_dir=root_dir+'grasp/braindecode/result/'
import os
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

sampling_rate=1000
#traindata, valdata, testdata = rawData2('raw',activeChannels,move2=False)  # (chns, 15000/15001, 118) (channels, time, trials)
traindata, valdata, testdata = raw_input(6,split=True,move2=True)
traindata = traindata.transpose(2, 0, 1)  # (118, 20, 15000) (trials,channels,  time)
valdata = valdata.transpose(2, 0, 1) # (8, 20, 15000)
testdata = testdata.transpose(2, 0, 1)  # (8, 20, 15000)

#traindata=np.load(tmp_dir+'traindata.npy')
#valdata=np.load(tmp_dir+'valdata.npy')
#testdata=np.load(tmp_dir+'testdata.npy')

trainx, trainy = traindata[:, :-2, :], traindata[:, -2, :] #-2 is real force, -1 is target
valx, valy = valdata[:, :-2, :], valdata[:, -2, :]
testx, testy = testdata[:, :-2, :], testdata[:, -2, :]

step=500 #ms
T=1000 #ms

dataset_train = SEEGDataset3D(trainx, trainy,T,step)
dataset_val = SEEGDataset3D(valx, valy,T,step)
dataset_test = SEEGDataset3D(testx, testy,T,step)

# Dataloader for training process
train_loader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=dataset_val, batch_size=1, pin_memory=False)
test_loader = DataLoader(dataset=dataset_test, batch_size=1, pin_memory=False)

chnNum=trainx.shape[1]
learning_rate=0.001
epochs=100
totalLen=trainx.shape[2] #ms
batch_size=int((totalLen-T)/step) # 280
convfeature=40
tkernelSize=200
avgpoolKernel=100
maxpoolKernel=3
maxpoolStride=3
blockKernelSize=10
dropout=0.5
Lambda = 1e-6

checkshape=torch.squeeze(next(iter(test_loader))[0])
length=checkshape.shape[2] # torch.Size([28, 90, 1000])

#shallowConv:def __int__(self,length, chnNum, convfeature,kernelSize,avgpoolKernel,dropout):
#deepConv: def __init__(self,length,chnNum,convfeature,kernelSize,maxpoolKernel,maxpoolStride,dropout):
net=deepConv(length, chnNum, convfeature, tkernelSize,blockKernelSize, maxpoolKernel,maxpoolStride,dropout)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

#checkpoint = torch.load('/Users/long/BCI/python_scripts/grasp/TSceptionWithoutMovement2/checkpoint20.pth')
#net.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
debugg=False
debugg=True
for epoch in range(epochs):
    print("------ epoch " + str(epoch) + " -----")
    net.train()

    loss_epoch=0
    #trial=0
    for trial, (trainx, target) in enumerate(train_loader): # ([1, 15000, 19]), ([1, 15000])
        if debugg==True: # just test one trial
            if trial == 1:
                break
                pass
        optimizer.zero_grad()
        print("Training on trial " + str(trial) + ".")

        y_pred = net(trainx)
        # regularization
        loss1 = criterion(torch.squeeze(y_pred), torch.squeeze(target))
        loss2 = regulization(net, Lambda)
        #loss3 = y_pred.cpu().detach().numpy()
        #loss3 = np.std(np.diff(loss3.reshape(-1)))
        loss=loss1+loss2 #+loss3*0.001
        loss.backward()
        optimizer.step()

        ls=loss1.item()
        loss_epoch+=ls
        with open(result_dir+"trainlose.txt", "a") as f:
            f.write(str(loss1) + "\n")
    print("----- Epoch "+str(epoch)+" loss:"+str(loss_epoch/(trial+1))+". -----")
    if epoch % 1 ==0:
        net.eval()
        with torch.no_grad():
            vpredAll = []
            vtargetAll = []
            for trial, (valx, vtarget) in enumerate(val_loader):  # ([1, 15000, 19]), ([1, 15000])
                print("Validating on trial " + str(trial) + ".")

                vpred = net(valx)
                vpredAll.append(vpred)
                vtargetAll.append(torch.squeeze(vtarget))
                loss3 = criterion(torch.squeeze(vpred), torch.squeeze(vtarget))
                with open(result_dir+"testlose.txt", "a") as f:
                    f.write(str(loss3) + "\n")
        vpredAll = np.concatenate(vpredAll,axis=0)
        vtargetAll = np.concatenate(vtargetAll, axis=0)

        fig, ax = plt.subplots(figsize=(6, 3))
        plt.ion()
        ax.clear()
        ax.plot(vtargetAll, label="True", linewidth=1)
        ax.plot(vpredAll, label='Predicted - Test', linewidth=1)
        ax.legend(loc='upper left')
        figname = result_dir+'prediction' + str(epoch) + '.pdf'
        fig.savefig(figname)
        plt.close(fig)
    if epoch % 5==0:
        state = {
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        savepath = result_dir+'checkpoint'+str(epoch)+'.pth'
        torch.save(state, savepath)
