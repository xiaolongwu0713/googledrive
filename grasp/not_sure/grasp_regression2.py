import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from d2l import torch as d2l
#channels=[8 9 10 18 19 20 21 22 23 24 62 63 69 70 105 107 108 109 110];
#trials=[1:20,31:40];

datafile1='/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TrainData.mat'
datafile2='/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TestData.mat'
raw1=scipy.io.loadmat(datafile1)
raw2=scipy.io.loadmat(datafile2)
train=raw1['train'] # (6299, 115)
test=raw2['test'] #(2699, 115)
tmp=np.concatenate((train,test),0) # (8998, 115)
x=torch.FloatTensor(tmp[:,0:-1]) #torch.Size([8998, 114])
y=torch.FloatTensor(tmp[:,-1]) #torch.Size([8998])

df=pd.DataFrame(data=tmp[0:,0:],index=[i for i in range(tmp.shape[0])],columns=['f'+str(i) for i in range(tmp.shape[1])])
df[('f'+str(i) for i in range(107,113))].describe()


######  linear regression, resemble the pls algrithom
x1=tmp[0:5000,0:-1]
y1=tmp[0:5000,-1]
x2=tmp[5000:,0:-1]
y2=tmp[5000:,-1]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
regressor = LinearRegression()
regressor.fit(x1,y1)
y_pred = regressor.predict(x2)
print(r2_score(y2,y_pred))
########

### test data
from scipy import signal
import matplotlib.pyplot as plt
t = np.linspace(0, 20, 10000, endpoint=False)
sig1=signal.square(2 * np.pi * 5 * t)[0:8998]
sig2=sig1
x=sig1
x=np.expand_dims(x, axis=1)
sig1=np.expand_dims(sig1, axis=1)
sig2=np.expand_dims(sig2, axis=1)
for i in range(113):
    x=np.concatenate((x,sig1*i*0.1),1)
y=x[:,1]*2+x[:,15]*3-x[:,10]+10*x[:,30]-100*x[:,100]
###

norm = MinMaxScaler().fit(x)
x = norm.transform(x)
x=torch.FloatTensor(x)
y=torch.FloatTensor(y)
target=y
xdf=pd.DataFrame(data=x[0:,0:],index=[i for i in range(x.shape[0])],columns=['f'+str(i) for i in range(x.shape[1])])


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

in_features=114
def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()

train_ls, test_ls = [], []
#train_iter = d2l.load_array((x.t(), y.t()), 299)

## train data
class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__(self):
        return len(self.X_data)
#train_data = trainData(torch.FloatTensor(trainx),torch.FloatTensor(trainy))
train_data = trainData(x,y)
train_loader = DataLoader(dataset=train_data, batch_size=299, shuffle=True)

#a,b=next(iter(train_iter))
net = get_net()
loss = nn.MSELoss()
learning_rate=0.1
weight_decay=0
num_epochs=100
optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate,weight_decay = weight_decay)
for epoch in range(num_epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        l = loss(net(X), y)
        l.backward()
        optimizer.step()
    train_ls.append(log_rmse(net, x, target))
    print(f'epoch: {epoch:3} loss: {l.item():10.8f}')
