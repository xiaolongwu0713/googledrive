import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from grasp.TSception.utils import SEEGDataset, regulization
from grasp.TSception.Models import TSception

# load the data
from grasp.utils import rawData

sampling_rate=1000
traindata, valdata, testdata = rawData()  # (20, 15000, 118) (channels, time, trials)
traindata = traindata.transpose(2, 0, 1)  # (118, 20, 15000) (trials,channels,  time)
valdata = valdata.transpose(2, 0, 1) # (8, 20, 15000)
testdata = testdata.transpose(2, 0, 1)  # (8, 20, 15000)
trainx, trainy = traindata[:, :-1, :], traindata[:, -1, :]
valx, valy = valdata[:, :-1, :], valdata[:, -1, :]
testx, testy = testdata[:, -1, :], testdata[:, -1, :]

dataset_train = SEEGDataset(trainx, trainy)
dataset_val = SEEGDataset(valx, valy)
dataset_test = SEEGDataset(testx, testy)

# Dataloader for training process
train_loader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=dataset_val, batch_size=1, pin_memory=False)
test_loader = DataLoader(dataset=dataset_test, batch_size=1, pin_memory=False)

chnNum=19
learning_rate=0.001
epochs=100
step=50 #ms
T=1000 #ms
totalLen=15000 #ms
batch_size=int((totalLen-T)/step) # 280
num_T = 3 # (6 conv2d layers) * ( 3 kernel each layer)
num_S = 3
hidden_size=222
dropout=0.2
Lambda = 1e-6

# __init__(self,input_size, sampling_rate, num_T, num_S, hiden, dropout_rate)
net = TSception(chnNum,sampling_rate, num_T, num_S, hidden_size, dropout).float()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


for epoch in range(epochs):
    print("------ epoch " + str(epoch) + " -----")
    net.train()

    loss_epoch=0
    #trial=0
    for trial, (trainx, trainy) in enumerate(train_loader): # ([1, 15000, 19]), ([1, 15000])
        # debug on first trail
        if trial == 1:
            #break
            pass
        optimizer.zero_grad()
        print("Training on trial " + str(trial) + ".")

        x = np.zeros((batch_size, 1, chnNum, T)) # 4D:(280,1,19,1000ms):(batch_size, planes, height, weight)
        target = np.zeros((batch_size,1)) # (280, 1)

        # format 1 trial into 3D tensor
        for bs in range(batch_size):
            x[bs, 0, :, :] = trainx[0, :, bs*step:(bs*step + T)]
            target[bs,0] = trainy[0,bs*step + T +1]
        y_pred = net(torch.from_numpy(x).float())
        y_true = torch.from_numpy(target)

        # regularization
        loss1 = criterion(y_pred, y_true.float())
        loss2 = regulization(net, Lambda)
        loss=loss1+loss2
        loss.backward()
        optimizer.step()

        ls=loss1.item()
        loss_epoch+=ls
        #print("Loss: " + str(ls) + ".")
        #trial+=1
    print("----- Epoch "+str(epoch)+" loss:"+str(loss_epoch/(trial+1))+". -----")
    if epoch % 1 ==0:
        net.eval()
        with torch.no_grad():
            vpred = []
            vtarget = []
            for trial, (valx, valy) in enumerate(val_loader):  # ([1, 15000, 19]), ([1, 15000])
                print("Validating on trial " + str(trial) + ".")

                vx = np.zeros((batch_size, 1, chnNum, T))  # 4D:(?,1,19,1000ms):(batch_size, planes, height, weight)
                target = np.zeros((batch_size, 1))

                # format 1 trial into 3D tensor
                for bs in range(batch_size):
                    vx[bs, 0, :, :] = valx[0, :, (bs * step):(bs * step + T)]
                    target[bs, 0] = valy[0,bs * step + T + 1]
                y_pred = net(torch.from_numpy(vx).float())
                vpred.append(y_pred)
                vtarget.append(target)
        vpred = np.concatenate(vpred,axis=0)
        vtarget = np.concatenate(vtarget, axis=0)

        fig, ax = plt.subplots(figsize=(6, 3))
        plt.ion()
        ax.clear()
        ax.plot(vtarget, label="True", linewidth=1)
        ax.plot(vpred, label='Predicted - Test', linewidth=1)
        ax.legend(loc='upper left')
        figname = '/Users/long/BCI/python_scripts/grasp/TSception/result/prediction' + str(epoch) + '.pdf'
        fig.savefig(figname)
        plt.close(fig)
