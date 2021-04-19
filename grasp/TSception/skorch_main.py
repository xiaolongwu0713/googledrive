import torch
from skorch.helper import predefined_split
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
import numpy as np
import matplotlib.pyplot as plt

from grasp.TSception.Models import TSception2
from grasp.utils import rawData2
from grasp.config import activeChannels, root_dir, tmp_dir


torch.manual_seed(0)
#torch.cuda.manual_seed(0)

result_dir=root_dir+'grasp/TSception/skorch/'
sampling_rate=1000
traindata, valdata, testdata = rawData2('band',activeChannels,move2=True)  # (chns, 15000/15001, 118) (channels, time, trials)
##traindata, valdata, testdata = rawData2('raw','all',move2=True)
traindata = traindata.transpose(2, 0, 1)  # (118, 20, 15000) (trials,channels,  time)
valdata = valdata.transpose(2, 0, 1) # (8, 20, 15000)
testdata = testdata.transpose(2, 0, 1)  # (8, 20, 15000)

np.save(tmp_dir+'traindata',traindata[:5,:,:])
np.save(tmp_dir+'valdata',valdata[:5,:,:])
np.save(tmp_dir+'testdata',testdata[:5,:,:])

#traindata=np.load(tmp_dir+'traindata.npy')
#valdata=np.load(tmp_dir+'valdata.npy')
#testdata=np.load(tmp_dir+'testdata.npy')


trainx, trainy = traindata[:, :-1, :], traindata[:, -1, :] #-2 is real force, -1 is target
valx, valy = valdata[:, :-1, :], valdata[:, -1, :]
testx, testy = testdata[:, :-1, :], testdata[:, -1, :]


# (10, 110, 15001)
samples=trainx.shape[0]
chnNum=trainx.shape[1]
step=50 #ms
T=1000 #ms
totalLen=trainx.shape[2] #ms
batch_size=int((totalLen-T)/step) # 280

xx=[]
yy=[]
for sample in range(samples):
    x = np.zeros((batch_size, 1, chnNum, T)) # 4D:(280,1,19,1000ms):(batch_size, planes, height, weight)
    targetd = np.zeros((batch_size,1)) # (280, 1)
    target = np.zeros((batch_size, 1))  # (280, 1)

    # format 1 trial into 3D tensor
    # result: regress to force derative not good at all
    for bs in range(batch_size):
        x[bs, 0, :, :] = trainx[0, :, bs*step:(bs*step + T)]
        target[bs, 0] = trainy[0, bs * step + T + 1] # force
    xx.append(x)
    yy.append(target)
xx=np.asarray(xx).astype(np.float32) # (118, 28, 1, 110, 1000)
yy=np.asarray(yy).astype(np.float32) # (118, 28, 1)

num_T = 3 # (6 conv2d layers) * ( 3 kernel each layer)
num_S = 3

model=TSception2(T, step, sampling_rate,chnNum, num_T, num_S,batch_size).float()

from skorch.callbacks import Callback

preds=[]
targets=[]
tracking=[]

class plotPrediction(Callback):
    def on_epoch_begin(self, net,dataset_train=None, dataset_valid=None, **kwargs):
        preds.clear()
        targets.clear()

    def on_batch_end(self, net, X=None, y=None, training=None, **kwargs):
        print('haha')
        if training==False:
            target=y.squeeze().numpy()
            step=kwargs
            loss=step['loss']
            y_pred=step['y_pred']
            y_pred=y_pred.squeeze().detach().numpy()
            preds.append(y_pred)
            targets.append(target)

    def on_epoch_end(self, net,dataset_train=None, dataset_valid=None, **kwargs):
        fig, ax = plt.subplots(figsize=(6, 3))
        plt.ion()
        ax.clear()
        a=np.concatenate(targets)
        b=np.concatenate(preds)
        ax.plot(a, label="True", linewidth=1)
        ax.plot(b, label='Predicted - Test', linewidth=1)
        ax.legend(loc='upper left')
        figname = result_dir + 'prediction' + str(len(net.history)) + '.png'
        fig.savefig(figname)
        plt.close(fig)

net = NeuralNetRegressor(
    model,
    #train_split=predefined_split(valid_set),
    iterator_train__shuffle=True,
    #train_split=5,
    max_epochs=2,
    lr=0.001,
    batch_size=1,
    optimizer=torch.optim.Adam,
    criterion = nn.MSELoss,
    callbacks=[('plotPrediction', plotPrediction()),],
    #device='cuda',  # uncomment this to train with CUDA
)
net.fit(xx, yy)