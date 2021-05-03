import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from torch.utils.data import DataLoader
from grasp.TSception.utils import regulization
from grasp.utils import SEEGDataset, freq_input, cuda_or_cup, set_random_seeds, savemode, loadmode
from grasp.MLP.model import MLP
from grasp.config import root_dir, tmp_dir

import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
result_dir=os.getcwd()+'/result/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

seed = 123456789  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed)
device=cuda_or_cup()
sid=6
#traindata, valdata, testdata = rawData2('raw',activeChannels,move2=False)  # (chns, 15000/15001, 118) (channels, time, trials)
traindata, valdata, testdata = freq_input(sid,split=True,move2=True)
traindata = traindata.transpose(2, 0, 1)  # (118, 20, 15000) (trials,channels,  time)
valdata = valdata.transpose(2, 0, 1) # (8, 20, 15000)
testdata = testdata.transpose(2, 0, 1)  # (8, 20, 15000)

#traindata=np.load(tmp_dir+'traindata.npy')
#valdata=np.load(tmp_dir+'valdata.npy')
#testdata=np.load(tmp_dir+'testdata.npy')

trainx, trainy = traindata[:, :-2, :], traindata[:, -2, :] #-2 is real force, -1 is target
valx, valy = valdata[:, :-2, :], valdata[:, -2, :]
testx, testy = testdata[:, :-2, :], testdata[:, -2, :]

dataset_train = SEEGDataset(trainx, trainy)
dataset_val = SEEGDataset(valx, valy)
dataset_test = SEEGDataset(testx, testy)

# Dataloader for training process
train_loader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=dataset_val, batch_size=1, pin_memory=False)
test_loader = DataLoader(dataset=dataset_test, batch_size=1, pin_memory=False)

chnNum=trainx.shape[1]
learning_rate=0.001
epochs=100
step=500 #ms
T=1000 #ms
totalLen=trainx.shape[2] #ms
batch_size=int((totalLen-T)/step) # 280
num_T = 3 # (6 conv2d layers) * ( 3 kernel each layer)
num_S = 3
hidden_size=222
dropout=0.2
Lambda = 1e-6

# __init__(self,input_size, sampling_rate, num_T, num_S, hiden, dropout_rate)
#net = IMVTensorLSTM(X_train.shape[2], 1, 128)
#net = IMVTensorLSTM(114, 1, 500)
net = MLP(chnNum).float()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
net, optimizer=loadmode(result_dir,99,net,optimizer)
#checkpoint = torch.load('/Users/long/BCI/python_scripts/grasp/TSceptionWithoutMovement2/checkpoint20.pth')
#net.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
debugg=False
#debugg=True
def train_model(epoch,train_dl, model,optimizer,criterion):
    model.train()
    #optimizer = SGD(model.parameters(), lr=learning_rate) # failed to converge
    # enumerate mini batches
    ls=.0
    for i, (inputs, targets) in enumerate(train_dl):
        yhat = model(inputs.float())
        loss = criterion(torch.squeeze(yhat), targets.float())
        ls=ls+loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_model(epoch,test_dl,model):
    model.eval()
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs.float())
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        #actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    predictions = np.concatenate(predictions)
    actuals=np.concatenate(actuals)
    mse = mean_squared_error(np.squeeze(actuals), np.squeeze(predictions))
    plt.cla()
    ax.plot(actuals, color="orange")
    ax.plot(predictions, color="blue")
    ax.set_title('Testing set,MSE='+str(mse), fontsize=25)
    #plt.show()
    #plt.pause(1)
    filename=result_dir+'test_epoch'+str(epoch)+'.png'
    plt.savefig(filename)
    return mse

fig, ax = plt.subplots(figsize=(12,7))
plt.ion()
epochs=100
for epoch in range(100,200):
    # train the model
    train_model(epoch,train_loader, net,optimizer,criterion)
    # evaluate the model
    if epoch % 1 == 0:
        print('evaluating')
        mse = evaluate_model(epoch,test_loader,net)
        savemode(result_dir,epoch,net,optimizer)