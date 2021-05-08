import torch
from skorch.callbacks import Callback
from skorch.helper import predefined_split, SliceDataset
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from grasp.TSception.Models import TSception2
from grasp.myskorch import MyRegressor, on_epoch_begin_callback, on_batch_end_callback, on_epoch_end_callback
from grasp.utils import freq_input, SEEGDataset, set_random_seeds, cuda_or_cup, windTo3D_x, windTo3D_y, SEEGDataset3D, \
    raw_input
from grasp.config import *


sid=2
device=cuda_or_cup()
seed = 123456789  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed)


result_dir=root_dir+'grasp/TSception/result_gs/'
import os
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
sampling_rate=1000

#Note: comment below to debug faster
#traindata, valdata, testdata = raw_input(sid,split=True,move2=True)  # (chns, 15000/15001, 118) (channels, time, trials)
##traindata, valdata, testdata = rawData2('raw','all',move2=True)
#traindata = traindata.transpose(2, 0, 1)  # (118, 20, 15000) (trials,channels,  time)
#valdata = valdata.transpose(2, 0, 1) # (8, 20, 15000)
#testdata = testdata.transpose(2, 0, 1)  # (8, 20, 15000)

#np.save(tmp_dir+'traindata',traindata[:5,:,:])
#np.save(tmp_dir+'valdata',valdata[:5,:,:])
#np.save(tmp_dir+'testdata',testdata[:5,:,:])

traindata=np.load(tmp_dir+'traindata.npy')
valdata=np.load(tmp_dir+'valdata.npy')
testdata=np.load(tmp_dir+'testdata.npy')

trainx, trainy = traindata[:, :-2, :], traindata[:, -2, :] #-2 is real force, -1 is target
valx, valy = valdata[:, :-2, :], valdata[:, -2, :]
testx, testy = testdata[:, :-2, :], testdata[:, -2, :]

train_and_valx=np.concatenate((trainx,valx),axis=0)
train_and_valy=np.concatenate((trainy,valy),axis=0)

#train_and_valx=trainx
#train_and_valy=trainy

step=1000 #ms
T=1000 #ms

train_and_val_ds = SEEGDataset3D(train_and_valx, train_and_valy,T, step)
train_ds = SEEGDataset3D(trainx, trainy,T, step)
val_ds = SEEGDataset3D(valx, valy,T, step)
test_ds = SEEGDataset3D(testx, testy,T, step)
train_and_valy=windTo3D_y(train_and_valy,T,step).astype(np.float32)
#Note: no dataset when gridsearch:  https://skorch.readthedocs.io/en/stable/user/FAQ.html#how-do-i-use-sklearn-gridseachcv-when-my-data-is-in-a-dataset
X_sl = SliceDataset(train_and_val_ds, idx=0)  # idx=0 is the default
y_sl = SliceDataset(train_and_val_ds, idx=1)

learning_rate=0.002
num_T = 3 # (6 conv2d layers) * ( 3 kernel each layer)
num_S = 3
num_ST=3
dropout=0.5
Lambda = 1e-6
samples=trainx.shape[0]
chnNum=trainx.shape[1]
totalLen=trainx.shape[2] #ms
batch_size=int((totalLen-T)/step) # 280

#Note: do not init here. Refer to doc: https://skorch.readthedocs.io/en/stable/user/neuralnet.html#module
#module=TSception2(T, step, sampling_rate,chnNum, num_T, num_S,batch_size,dropout).float()
from skorch.callbacks import Checkpoint, TrainEndCheckpoint, EarlyStopping
cp = Checkpoint(dirname=result_dir) # Will automatically save the model when there is an improvement during validation.
train_end_cp = TrainEndCheckpoint(dirname=result_dir) # save model and the end of the training

#early_stopping_callbacks = EarlyStopping(monitor='valid_loss', lower_is_better=True, patience=5)
#__init__(self, chnNum, sampling_rate, num_T, num_S,dropout):
net = MyRegressor(
    module=TSception2,
    module__chnNum=chnNum,
    module__sampling_rate=sampling_rate,
    module__num_T=num_ST,
    module__num_S=num_ST,
    module__dropout=dropout,
    lambda1=Lambda, # can be at at any sequence location.
    iterator_train__shuffle=True,

    #train_split=predefined_split(val_ds), #disable this when do grid search
    train_split=None, # set to be None when doing grid search
    verbose=0, # grid search

    max_epochs=2,
    lr=learning_rate,
    batch_size=1,
    optimizer=torch.optim.Adam,
    criterion = nn.MSELoss,
    callbacks=[('on_epoch_begin_callback', on_epoch_begin_callback),('on_batch_end_callback',on_batch_end_callback),
               ],# ('on_epoch_end_callback', on_epoch_end_callback(result_dir)),
               #cp,train_end_cp], # early_stopping_callbacks
    device = device
)

params = {
    'lr': [0.001, 0.003],
    #'module__num_S':[2,3],
    #'lambda1':[1e-6,1e-5],
    #'module__num_T':[2,3,4,5,6],
    #'module__dropout':[0.1,0.3,0.5,0.7],
    #'optimizer': [torch.optim.Adam, torch.optim.Adagrad ,torch.optim.SGD],
    #'optimizer__weight_decay':[0, 1e-04, 1e-03],
    #'max_epochs': [100,200, 300],
    #'module__num_units': [10, 20],
}
#net.fit(train_and_val_ds,y=None)
#net.fit(X_sl,y_sl)
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(net, params, refit=False, cv=2, scoring='neg_mean_squared_error', verbose=2) # score: higher is better
gs.fit(X_sl,train_and_valy)

print(gs.best_score_, gs.best_params_)

lr_fitted=gs.best_params_['lr']
#dropout_fitted=gs.best_params_['module__dropout']

net2 = MyRegressor(
    module=TSception2,
    module__chnNum=chnNum,
    module__sampling_rate=sampling_rate,
    module__num_T=num_ST,
    module__num_S=num_ST,
    module__dropout=dropout,
    lambda1=Lambda, # can be at at any sequence location.
    iterator_train__shuffle=True,

    train_split=predefined_split(val_ds), #disable this when do grid search
    #train_split=None, # set to be None when doing grid search
    #verbose=0, # set to 0 when grid search

    max_epochs=2,
    lr=lr_fitted,
    batch_size=1,
    optimizer=torch.optim.Adam,
    criterion = nn.MSELoss,
    callbacks=[('on_epoch_begin_callback', on_epoch_begin_callback),('on_batch_end_callback',on_batch_end_callback),
               ('on_epoch_end_callback', on_epoch_end_callback(result_dir)),],
               #cp,train_end_cp], # early_stopping_callbacks
    device = device
)
net2.fit(train_ds,y=None)
