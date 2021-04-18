import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
#channels=[8 9 10 18 19 20 21 22 23 24 62 63 69 70 105 107 108 109 110];
#trials=[1:20,31:40];

datafile='/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TrainData.mat'
dataset_raw = scipy.io.loadmat(datafile)
dataset=dataset_raw['train'] # numpy array (114, 299, 30)
target=dataset_raw['trainForce'] # (299, 30)

train=dataset[:,:,0:29] # (114, 299, 29)
test=dataset[:,:,-1] # (114, 299)



train = torch.FloatTensor(np.transpose(train, (2,0,1))) # torch.Size([29, 114, 299])
test=torch.FloatTensor(np.transpose(test,(1,0))) # (114, 299)
targets = torch.FloatTensor(np.transpose(target, (1,0))) #  torch.Size([30, 299])
test_target=targets[-1,:] # torch.Size([299])
trials=train.shape[0]

num_layer=2

class LSTM(nn.Module):
    def __init__(self, input_size=114, hidden_size=200, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,num_layer)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_cell = (torch.zeros(num_layer,1,self.hidden_size),torch.zeros(num_layer,1,self.hidden_size))
        self.relu = nn.ReLU()
    def forward(self, input_seq):
        outputs, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.relu(self.linear(outputs.squeeze()))
        return predictions

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    model.train()
    loss = 0
    #model.hidden_cell = (torch.zeros(1, 1, model.hidden_size), torch.zeros(1, 1, model.hidden_size))
    for trial in range(trials):
        input=np.transpose(train[trial,:,:],(1,0)).unsqueeze(1) # torch.Size([299, 1, 114])
        target=targets[trial,:]

        model.hidden_cell = (torch.zeros(num_layer, 1, model.hidden_size),torch.zeros(num_layer, 1, model.hidden_size))
        y_hat=model(input)
        loss = loss_function(y_hat.squeeze(), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_test_predict = model(test.unsqueeze(1))
        losst = loss_function(y_test_predict.squeeze(), test_target)

    #print(f'epoch: {epoch:3} loss: {loss.item():10.8f}, test loss: {losst.item():10.8f}')
    print(f'epoch: {epoch:3} loss: {loss.item():10.8f}, test loss: {losst.item():10.8f}')

y_train=y_hat.detach().numpy().squeeze()
y=y_test_predict.numpy().squeeze()

