import sys
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
from grasp.utils import read_fbanddata
from grasp.NarxModelNoForceFeedback.model import DA_RNN

# trainx: (180, 299, 133), trainy:(299, 133)
T=20
trainx,trainy,testx,testy = read_fbanddata()
chnNum=trainx.shape[0]
totalLen=trainx.shape[1]
hiddenSize_encoder = chnNum
hiddenSize_decoder = chnNum
batch_size=totalLen - T

def test(testx, testy, model):
    model.eval()
    y_preds = []
    with torch.no_grad():
        for trial in range(testx.shape[2]):
            print("Testing on trial " + str(trial) + ".")
            trialdata = testx[:, :, trial]
            targetdata = testy[:, trial]

            x = np.zeros((batch_size, T, chnNum))
            y = np.zeros((batch_size, T))  # 2D history y
            target = targetdata[T:]  # (279,)

            # format 1 trial into 3D tensor
            for bs in range(batch_size):
                x[bs, :, :] = trialdata[:, bs:bs + T].T
                y[bs, :] = targetdata[bs:bs + T]
            # x = x.swapaxes(1, 2) #(batch_size,timeT,channel)
            y_pred = net(x) # (279,)
            y_preds.append(y_pred)
    return y_preds

lr=0.001
epochs=100
net = DA_RNN(T,chnNum,hiddenSize_encoder,hiddenSize_decoder,lr,epochs)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(epochs):
    print("------ epoch " + str(epoch) + " -----")
    net.train()
    # training on all trials
    # pick some trial. movw1|move2|move3|move4: 0-33|-68|-96|-132|
    picktrials = np.concatenate((np.arange(0,9),np.arange(40,49),np.arange(70,79),np.arange(100,109)),axis=0)
    #picktrials = np.arange(0,2)
    #for trial in range(trainx.shape[2]-130): # too slow to train on 133 trials
    for trial in picktrials:
        optimizer.zero_grad()
        print("Training on trial " + str(trial) + ".")
        trialdata = trainx[:, :, trial]
        targetdata = trainy[:, trial]

        x = np.zeros((batch_size, T, chnNum))
        y = np.zeros((batch_size, T)) # 2D history y
        target = targetdata[T:]  # 1D target

        # format 1 trial into 3D tensor
        for bs in range(batch_size):
            x[bs, :, :] = trialdata[:, bs:bs + T].T
            y[bs,:] = targetdata[bs:bs + T]
        # x = x.swapaxes(1, 2) #(batch_size,timeT,channel)
        y_pred = net(x)
        y_true = torch.from_numpy(target)
        loss = criterion(y_pred, y_true.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        ls=loss.item()
        print("Loss: " + str(ls) + ".")
    if epoch % 1 ==0:
        testy_pred = test(testx,testy,net) # testx: (180, 299, 8), testy:(299, 8)
        padding = np.zeros((T,1))
        # padding T before prediction
        testy_pred = [np.concatenate((padding,sublist),axis=0) for sublist in testy_pred]
        testy_pred = [signalpoint for sublist in testy_pred for signalpoint in sublist]
        testy_flatten = np.reshape(testy.T, (-1, 1))
        fig, ax = plt.subplots(figsize=(6, 3))
        plt.ion()
        ax.clear()
        ax.plot(testy_flatten, label="True", linewidth=0.1)
        ax.plot(testy_pred, label='Predicted - Test', linewidth=0.1)
        ax.legend(loc='upper left')
        figname = '/Users/long/BCI/python_scripts/grasp/NarxModelNoForceFeedback/result/prediction' + str(epoch) + '.pdf'
        fig.savefig(figname)
        plt.close(fig)
