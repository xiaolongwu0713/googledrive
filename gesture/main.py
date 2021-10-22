#%cd /content/drive/MyDrive/
# raw_data is imported from global config

#%%capture
#! pip install hdf5storage
#! pip install mne==0.23.0
#! pip install torch
#! pip install Braindecode==0.5.1
#! pip install timm

import os, re
import hdf5storage
import numpy as np
from scipy.io import savemat
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from braindecode.datautil import (create_from_mne_raw, create_from_mne_epochs)
import torch
import random
from common_dl import set_random_seeds
from common_dl import myDataset
from comm_utils import slide_epochs
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from braindecode.models import ShallowFBCSPNet,EEGNetv4,Deep4Net
from gesture.models.deepmodel import deepnet,deepnet_resnet
from gesture.models.d2l_resnet import d2lresnet
from gesture.models.EEGModels import DeepConvNet_210519_512_10
from gesture.models.tsception import TSception

from gesture.myskorch import on_epoch_begin_callback, on_batch_end_callback
from gesture.config import *
from gesture.preprocess.chn_settings import get_channel_setting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

import inspect as i
import sys
#sys.stdout.write(i.getsource(deepnet))

sid=10 #4
class_number=5
Session_num,UseChn,EmgChn,TrigChn, activeChan = get_channel_setting(sid)
#fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]
fs=1000

project_dir=data_dir+'preprocessing'+'/P'+str(sid)+'/'
model_path=project_dir + 'pth' +'/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == sid][0]

loadPath = data_dir+'preprocessing'+'/P'+str(sid)+'/preprocessing2.mat'
mat=hdf5storage.loadmat(loadPath)
data = mat['Datacell']
channelNum=int(mat['channelNum'][0,0])
data=np.concatenate((data[0,0],data[0,1]),0)
del mat
# standardization
# no effect. why?
if 1==0:
    chn_data=data[:,-3:]
    data=data[:,:-3]
    scaler = StandardScaler()
    scaler.fit(data)
    data=scaler.transform((data))
    data=np.concatenate((data,chn_data),axis=1)

# stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
chn_names=np.append(["seeg"]*len(UseChn),["stim0", "emg","stim1"])
chn_types=np.append(["seeg"]*len(UseChn),["stim", "emg","stim"])
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
raw = mne.io.RawArray(data.transpose(), info)


# gesture/events type: 1,2,3,4,5
events0 = mne.find_events(raw, stim_channel='stim0')
events1 = mne.find_events(raw, stim_channel='stim1')
# events number should start from 0: 0,1,2,3,4, instead of 1,2,3,4,5
events0=events0-[0,0,1]
events1=events1-[0,0,1]

#print(events[:5])  # show the first 5
# Epoch from 4s before(idle) until 4s after(movement) stim1.
raw=raw.pick(["seeg"])
epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)
# or epoch from 0s to 4s which only contain movement data.
# epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

epoch1=epochs['0'].get_data() # 20 trials. 8001 time points per trial for 8s.
epoch2=epochs['1'].get_data()
epoch3=epochs['2'].get_data()
epoch4=epochs['3'].get_data()
epoch5=epochs['4'].get_data()
list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]
total_len=list_of_epochs[0].shape[2]



# validate=test=2 trials
trial_number=[list(range(epochi.shape[0])) for epochi in list_of_epochs] #[ [0,1,2,...19],[0,1,2...19],... ]
test_trials=[random.sample(epochi, 2) for epochi in trial_number]
# len(test_trials[0]) # test trials number
trial_number_left=[np.setdiff1d(trial_number[i],test_trials[i]) for i in range(class_number)]

val_trials=[random.sample(list(epochi), 2) for epochi in trial_number_left]
train_trials=[np.setdiff1d(trial_number_left[i],val_trials[i]).tolist() for i in range(class_number)]

# no missing trials
assert [sorted(test_trials[i]+val_trials[i]+train_trials[i]) for i in range(class_number)] == trial_number

test_epochs=[epochi[test_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)] # [ epoch0,epoch1,epch2,epoch3,epoch4 ]
val_epochs=[epochi[val_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)]
train_epochs=[epochi[train_trials[clas],:,:] for clas,epochi in enumerate(list_of_epochs)]


wind=500
stride=500
X_train=[]
y_train=[]
X_val=[]
y_val=[]
X_test=[]
y_test=[]

for clas, epochi in enumerate(test_epochs):
    Xi,y=slide_epochs(epochi,clas,wind, stride)
    assert Xi.shape[0]==len(y)
    X_test.append(Xi)
    y_test.append(y)
X_test=np.concatenate(X_test,axis=0) # (1300, 63, 500)
y_test=np.asarray(y_test)
y_test=np.reshape(y_test,(-1,1)) # (5, 270)

for clas, epochi in enumerate(val_epochs):
    Xi,y=slide_epochs(epochi,clas,wind, stride)
    assert Xi.shape[0]==len(y)
    X_val.append(Xi)
    y_val.append(y)
X_val=np.concatenate(X_val,axis=0) # (1300, 63, 500)
y_val=np.asarray(y_val)
y_val=np.reshape(y_val,(-1,1)) # (5, 270)

for clas, epochi in enumerate(train_epochs):
    Xi,y=slide_epochs(epochi,clas,wind, stride)
    assert Xi.shape[0]==len(y)
    X_train.append(Xi)
    y_train.append(y)
X_train=np.concatenate(X_train,axis=0) # (1300, 63, 500)
y_train=np.asarray(y_train)
y_train=np.reshape(y_train,(-1,1)) # (5, 270)
chn_num=X_train.shape[1]

train_set=myDataset(X_train,y_train)
val_set=myDataset(X_val,y_val)
test_set=myDataset(X_test,y_test)

batch_size = 32
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)

train_size=len(train_loader.dataset)
val_size=len(val_loader.dataset)
test_size=len(test_loader.dataset)

# These values we found good for shallow network:
lr = 0.0001
weight_decay = 1e-10
batch_size = 32
n_epochs = 200

#one_window.shape : (208, 500)

# Extract number of chans and time steps from dataset
one_window=next(iter(train_set))[0]
n_chans = one_window.shape[0]


#net = ShallowFBCSPNet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',) # 51%
#net = EEGNetv4(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)

net = deepnet(n_chans,class_number,input_window_samples=wind,final_conv_length='auto',) # 81%

#net = deepnet_resnet(n_chans,n_classes,input_window_samples=input_window_samples,expand=True) # 50%

#net=d2lresnet() # 92%

#net=TSception(208)

#net=TSception(1000,n_chans,3,3,0.5)

img_size=[n_chans,wind]
#net = timm.create_model('visformer_tiny',num_classes=n_classes,in_chans=1,img_size=img_size)
if cuda:
    net.cuda()

lr = 0.05
weight_decay = 1e-10

criterion = torch.nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)
#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# Decay LR by a factor of 0.1 every 7 epochs
lr_schedulerr = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

epoch_num = 100

for epoch in range(epoch_num):
    print("------ epoch " + str(epoch) + " -----")
    net.train()

    loss_epoch = 0

    running_loss = 0.0
    running_corrects = 0
    for batch, (trainx, trainy) in enumerate(train_loader):
        if isinstance(net, timm.models.visformer.Visformer):
            trainx=torch.unsqueeze(trainx,dim=1)
        optimizer.zero_grad()
        if (cuda):
            trainx = trainx.float().cuda()
        else:
            trainx = trainx.float()
        y_pred = net(trainx)
        #print("y_pred shape: " + str(y_pred.shape))
        preds = y_pred.argmax(dim=1, keepdim=True)
        #_, preds = torch.max(y_pred, 1)

        if cuda:
            loss = criterion(y_pred, trainy.squeeze().cuda().long())
        else:
            loss = criterion(y_pred, trainy.squeeze())

        loss.backward()  # calculate the gradient and store in .grad attribute.
        optimizer.step()
        running_loss += loss.item() * trainx.shape[0]
        running_corrects += torch.sum(preds.cpu().squeeze() == trainy.squeeze())
    #print("train_size: " + str(train_size))
    lr_schedulerr.step() # test it
    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size
    print("Training loss: {:.2f}; Accuracy: {:.2f}.".format(epoch_loss,epoch_acc.item()))
    #print("Training " + str(epoch) + ": loss: " + str(epoch_loss) + "," + "Accuracy: " + str(epoch_acc.item()) + ".")

    state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
             'epoch': epoch,
             'loss': epoch_loss
        }
    savepath = model_path + 'checkpoint' + str(epoch) + '.pth'
    #torch.save(state, savepath)
    running_loss = 0.0
    running_corrects = 0
    if epoch % 1 == 0:
        net.eval()
        # print("Validating...")
        with torch.no_grad():
            for _, (val_x, val_y) in enumerate(val_loader):
                if isinstance(net, timm.models.visformer.Visformer):
                    val_x = torch.unsqueeze(val_x, dim=1)
                if (cuda):
                    val_x = val_x.float().cuda()
                    # val_y = val_y.float().cuda()
                else:
                    val_x = val_x.float()
                    # val_y = val_y.float()
                outputs = net(val_x)
                #_, preds = torch.max(outputs, 1)
                preds = outputs.argmax(dim=1, keepdim=True)

                running_corrects += torch.sum(preds.cpu().squeeze() == val_y.squeeze())

        epoch_acc = running_corrects.double() / val_size
        print("Evaluation accuracy: {:.2f}.".format(epoch_acc.item()))



