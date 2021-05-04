import torch
from skorch.callbacks import Callback
from skorch.helper import predefined_split, SliceDataset
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from grasp.TSception.Models import TSception2
from grasp.myskorch import  MyRegressor, on_epoch_begin_callback, on_batch_end_callback
from grasp.utils import freq_input, SEEGDataset, set_random_seeds, cuda_or_cup, windTo3D_x, windTo3D_y
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

#Note: comment below to debug faster
#traindata, valdata, testdata = freq_input(sid,split=True,move2=True)  # (chns, 15000/15001, 118) (channels, time, trials)
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

learning_rate=0.002
num_T = 3 # (6 conv2d layers) * ( 3 kernel each layer)
num_S = 3
num_ST=3
dropout=0.5
Lambda = 1e-6
samples=trainx.shape[0]
chnNum=trainx.shape[1]
step=50 #ms
T=1000 #ms
totalLen=trainx.shape[2] #ms
batch_size=int((totalLen-T)/step) # 280


train_and_val_wind_x=[]
train_and_val_wind_y=[]
for trial in range(train_and_valx.shape[0]):
    train_and_val_wind_x.append([])
    train_and_val_wind_y.append([])
    train_and_val_wind_x[trial]=windTo3D_x(train_and_valx[trial],T,step)
    train_and_val_wind_y[trial]=windTo3D_y(train_and_valy[trial],T,step)

#train_and_val_final_x=np.concatenate( train_and_val_wind_x, axis=0 )
#train_and_val_final_y=np.concatenate( train_and_val_wind_y, axis=0 )

train_and_val_final_x=np.stack( train_and_val_wind_x).astype(np.float32) # (10, 280, 1, 90, 1000)
train_and_val_final_y=np.stack( train_and_val_wind_y).astype(np.float32) # (10, 280, 1)

#train_and_val_ds = SEEGDataset(train_and_valx, train_and_valy,T, step)
#val_ds = SEEGDataset(valx, valy,T, step)
#test_ds = SEEGDataset(testx, testy,T, step)

#Note: no dataset when gridsearch:  https://skorch.readthedocs.io/en/stable/user/FAQ.html#how-do-i-use-sklearn-gridseachcv-when-my-data-is-in-a-dataset
#X_sl = SliceDataset(train_and_val_ds, idx=0)  # idx=0 is the default
#y_sl = SliceDataset(train_and_val_ds, idx=1)


#Note: do not init here. Refer to doc: https://skorch.readthedocs.io/en/stable/user/neuralnet.html#module
module=TSception2(T, step, sampling_rate,chnNum, num_T, num_S,batch_size,dropout).float()
from skorch.callbacks import Checkpoint, TrainEndCheckpoint, EarlyStopping
cp = Checkpoint(dirname=result_dir) # Will automatically save the model when there is an improvement during validation.
train_end_cp = TrainEndCheckpoint(dirname=result_dir) # save model and the end of the training

early_stopping_callbacks = EarlyStopping(monitor='valid_loss', lower_is_better=True, patience=5)

net = MyRegressor(
    module=TSception2,
    module__wind_size=T,
    module__step=step,
    module__sampling_rate=sampling_rate,
    module__chnNum=chnNum,
    module__num_T=num_ST,
    module__num_S=num_ST,
    module__batch_size=batch_size,
    module__dropout=dropout,
    lambda1=Lambda, # can be at at sequence location.
    iterator_train__shuffle=True,
    #train_split=predefined_split(val_ds), #disable this when do grid search
    train_split=None, # grid search
    verbose=0, # grid search
    max_epochs=2,
    lr=learning_rate,
    batch_size=1,
    optimizer=torch.optim.Adam,
    criterion = nn.MSELoss,
    callbacks=[('on_epoch_begin_callback', on_epoch_begin_callback),
               ('on_batch_end_callback',on_batch_end_callback),],
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
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(net, params, refit=False, cv=2, scoring='neg_mean_squared_error', verbose=2) # score: higher is better
gs.fit(train_and_val_final_x,train_and_val_final_y)




