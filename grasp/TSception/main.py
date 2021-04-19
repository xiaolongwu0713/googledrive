import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from grasp.TSception.utils import regulization
from grasp.utils import SEEGDataset
from grasp.TSception.Models import TSception2
from examples.IMV_LSTM.networks import IMVTensorLSTM
# load the data: regression to target force derivative
from grasp.utils import rawData2
from grasp.config import activeChannels, root_dir

result_dir=root_dir+'grasp/TSception/resultBandInput/'
sampling_rate=1000
#traindata, valdata, testdata = rawData2('raw',activeChannels,move2=False)  # (chns, 15000/15001, 118) (channels, time, trials)
traindata, valdata, testdata = rawData2('raw','all',move2=True)
traindata = traindata.transpose(2, 0, 1)  # (118, 20, 15000) (trials,channels,  time)
valdata = valdata.transpose(2, 0, 1) # (8, 20, 15000)
testdata = testdata.transpose(2, 0, 1)  # (8, 20, 15000)
trainx, trainy = traindata[:, :-2, :], traindata[:, -2, :] #-2 is real force, -1 is target
valx, valy = valdata[:, :-2, :], valdata[:, -2, :]
testx, testy = testdata[:, :-2, :], testdata[:, -2, :]

chnNum=trainx.shape[1]
step=500 #ms
T=1000 #ms
totalLen=trainx.shape[2] #ms
batch_size=int((totalLen-T)/step) # 280

dataset_train = SEEGDataset(trainx, trainy,T, step)
dataset_val = SEEGDataset(valx, valy,T, step)
dataset_test = SEEGDataset(testx, testy,T, step)

# Dataloader for training process
train_loader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=dataset_val, batch_size=1, pin_memory=False)
test_loader = DataLoader(dataset=dataset_test, batch_size=1, pin_memory=False)


learning_rate=0.001
epochs=100

num_T = 3 # (6 conv2d layers) * ( 3 kernel each layer)
num_S = 3
hidden_size=222
dropout=0.2
Lambda = 1e-6

# __init__(self,input_size, sampling_rate, num_T, num_S, hiden, dropout_rate)
#net = IMVTensorLSTM(X_train.shape[2], 1, 128)
#net = IMVTensorLSTM(114, 1, 500)
net = TSception2(T, step, sampling_rate,chnNum, num_T, num_S,batch_size).float()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

#checkpoint = torch.load('/Users/long/BCI/python_scripts/grasp/TSceptionWithoutMovement2/checkpoint20.pth')
#net.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
debugg=False
#debugg=True
for epoch in range(epochs):
    print("------ epoch " + str(epoch) + " -----")
    net.train()

    loss_epoch=0
    #trial=0
    for trial, (trainx, trainy) in enumerate(train_loader): # ([1, 15000, 19]), ([1, 15000])
        trainy[0,-1]+=0.05
        if debugg==True: # just test one trial
            if trial == 1:
                break
                pass
        optimizer.zero_grad()
        print("Training on trial " + str(trial) + ".")

        x = np.zeros((batch_size, 1, chnNum, T)) # 4D:(280,1,19,1000ms):(batch_size, planes, height, weight)
        targetd = np.zeros((batch_size,1)) # (280, 1)
        target = np.zeros((batch_size, 1))  # (280, 1)

        # format 1 trial into 3D tensor
        # result: regress to force derative not good at all
        for bs in range(batch_size):
            x[bs, 0, :, :] = trainx[0, :, bs*step:(bs*step + T)]
            target[bs, 0] = trainy[0, bs * step + T + 1] # force
            targetd[bs,0] = abs(trainy[0,bs*step + T +1] - trainy[0,bs*step + T -50])*10+0.05 # force derative
        targetd[:,0] = [abs(item) / 5 if abs(item) > 0.2 else abs(item) for item in targetd[:,0]]
        y_pred = net(torch.from_numpy(x).float())
        #target = torch.from_numpy(target)

        # regularization
        loss1 = criterion(y_pred, torch.from_numpy(target))
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
            for trial, (valx, valy) in enumerate(val_loader):  # ([1, 15000, 19]), ([1, 15000])
                valy[-1]+=0.05
                print("Validating on trial " + str(trial) + ".")

                vx = np.zeros((batch_size, 1, chnNum, T))  # 4D:(?,1,19,1000ms):(batch_size, planes, height, weight)
                vtarget = np.zeros((batch_size, 1))
                vtargetd = np.zeros((batch_size, 1))

                # format 1 trial into 3D tensor
                for bs in range(batch_size):
                    vx[bs, 0, :, :] = valx[0, :, (bs * step):(bs * step + T)]
                    vtarget[bs, 0] = valy[0, bs * step + T + 1]
                    vtargetd[bs, 0] = abs(valy[0,bs * step + T + 1] - valy[0,bs * step + T - 50])*10 + 0.05
                vtargetd[:, 0] = [abs(item) / 5 if abs(item) > 0.5 else abs(item) for item in vtargetd[:, 0]]
                vtarget = torch.from_numpy(vtarget)
                y_pred = net(torch.from_numpy(vx.squeeze().transpose(0,2,1)).float())
                y_pred=np.expand_dims(y_pred, axis=1)
                vpredAll.append(y_pred)
                vtargetAll.append(vtarget)
                loss3 = criterion(y_pred, vtarget.float())
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
