#%cd /content/drive/MyDrive/

#%%capture
#! pip install hdf5storage
#! pip install mne==0.23.0
#! pip install torch==1.7.0
#! pip install Braindecode==0.5.1
#! pip install timm

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import mne
import torch
from torch.optim import lr_scheduler
from torch import nn
import timm
from common_dl import myDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from common_dl import set_random_seeds
from common_dsp import *
from gesture.models.d2l_resnet import d2lresnet
from myskorch import on_epoch_begin_callback, on_batch_end_callback
from ecog_finger.config import *
from ecog_finger.preprocess.chn_settings import  get_channel_setting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


try:
    mne.set_config('MNE_LOGGING_LEVEL','ERROR')
except TypeError as err:
    print(err)

sid=2
fs=1000
use_active_only=False
if use_active_only:
    active_chn=get_channel_setting(sid)
else:
    active_chn='all'

if 1==2:
    filename=data_dir+'fingerflex/data/'+str(sid)+'/'+str(sid)+'_fingerflex.mat'
    mat=scipy.io.loadmat(filename)
    data=mat['data'] # (46, 610040)

    if 1==1:
        scaler = StandardScaler()
        scaler.fit(data)
        data=scaler.transform((data))
    data=np.transpose(data)
    chn_num=data.shape[0]
    flex=np.transpose(mat['flex']) #(5, 610040)
    cue=np.transpose(mat['cue']) # (1, 610040)
    data=np.concatenate((data,cue),axis=0) # (47, 610040) / (47, 610040)
    chn_names=np.append(["ecog"]*chn_num,["stim"])  #,"thumb","index","middle","ring","little"])
    chn_types=np.append(["ecog"]*chn_num,["stim"])  #, "emg","emg","emg","emg","emg"])
    info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
    raw = mne.io.RawArray(data, info)

    events = mne.find_events(raw, stim_channel='stim')
    events=events-[0,0,1] #(150, 3)
    raw=raw.pick(picks=['ecog'])
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2,baseline=None)
    # or epoch from 0s to 4s which only contain movement data.
    # epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

    epoch1=epochs['0'].get_data() # 20 trials. 8001 time points per trial for 8s.
    epoch2=epochs['1'].get_data()
    epoch3=epochs['2'].get_data()
    epoch4=epochs['3'].get_data()
    epoch5=epochs['4'].get_data()
    list_of_epochs = [epoch1, epoch2, epoch3, epoch4, epoch5]


input='rawAndbands'
if input=='raw':
    filename=data_dir+'fingerflex/data/'+str(sid)+'/'+str(sid)+'_fingerflex.mat'
    mat=scipy.io.loadmat(filename)
    data=mat['data'] # (46, 610040)
    data=data[:,:-1]

    if 1==1:
        scaler = StandardScaler()
        scaler.fit(data)
        data=scaler.transform((data))
    data=np.transpose(data)
    chn_num=data.shape[0]
    flex=np.transpose(mat['flex']) #(5, 610040)
    cue=np.transpose(mat['cue']) # (1, 610040)
    data=np.concatenate((data,cue),axis=0) # (47, 610040) / (47, 610040)

    chn_names=np.append(["ecog"]*chn_num,["stim"])  #,"thumb","index","middle","ring","little"])
    chn_types=np.append(["ecog"]*chn_num,["stim"])  #, "emg","emg","emg","emg","emg"])
    info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
    raw = mne.io.RawArray(data, info)

    events = mne.find_events(raw, stim_channel='stim')
    events=events-[0,0,1] #(150, 3)
    raw=raw.pick(picks=['ecog'])


    epochs = mne.Epochs(raw, events, tmin=0, tmax=2,baseline=None)
    # or epoch from 0s to 4s which only contain movement data.
    # epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

    epoch1=epochs['0'].get_data() # 20 trials. 8001 time points per trial for 8s.
    epoch2=epochs['1'].get_data()
    epoch3=epochs['2'].get_data()
    epoch4=epochs['3'].get_data()
    epoch5=epochs['4'].get_data()
    list_of_epochs = [epoch1, epoch2, epoch3, epoch4, epoch5]

else:
    list_of_epochs=[]
    save_to = data_dir + 'fingerflex/data/' + str(sid) + '/'
    for fingeri in range(5):
        tmp = mne.read_epochs(save_to + 'rawBandEpoch'+str(fingeri)+'.fif')
        list_of_epochs.append(tmp.get_data())


wind=500
stride=50
s=0
X_train=[]
X_test=[]
labels_train=[]
labels_test=[]
total_len=list_of_epochs[0].shape[2]
labels=[]

test_number=10
for i in range(5):
    Xi_train = []
    Xi_test = []
    Xi=[]
    #ss=(total_len-wind)//stride
    for trial in list_of_epochs[i]: # (63, 2001)
        #trial=np.concatenate((trial,trial[-3:,:]),axis=0)
        s = 0
        while stride*s+wind<total_len:
            start=s * stride
            tmp=trial[:,start:(start+wind)]
            Xi.append(tmp)
            s=s+1
        # add the last window
        last_s=s-1
        if stride * last_s + wind<total_len-100:
            tmp=trial[:,-wind:]
            Xi.append(tmp)
    Xi_train=np.asarray(Xi[:-test_number]) # (260, 63, 500)
    Xi_test = np.asarray(Xi[-test_number:]) # (10, 63, 500)

    X_train.append(Xi_train)
    X_test.append(Xi_test)

    samples_number=len(Xi)
    label=[i]*samples_number
    label_train=label[:-test_number]
    label_test=label[-test_number:]

    labels_train.append(label_train)
    labels_test.append(label_test)


X_train=np.concatenate(X_train,axis=0) # (1300, 63, 500)
X_test=np.concatenate(X_test,axis=0) # (50, 63, 500)

labels_train=np.asarray(labels_train)
labels_train=np.reshape(labels_train,(-1,1)) # (5, 270)
labels_test=np.asarray(labels_test)
labels_test=np.reshape(labels_test,(-1,1)) # (5, 270)

# (871, 63, 500)/ (429, 63, 500)
X_train, X_val, y_train, y_val = train_test_split(X_train, labels_train, test_size=0.33, random_state=42)

train_set=myDataset(X_train,y_train)
val_set=myDataset(X_val,y_val)

batch_size = 32
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)

train_size=len(train_loader.dataset)
val_size=len(val_loader.dataset)

train_size

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed)

#net=d2lresnet()
net = timm.create_model('visformer_tiny',num_classes=5,in_chans=1,img_size=[496,500])
net = net.to(device)

lr = 0.05
weight_decay = 1e-10
epoch_num = 500

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
optimizer = torch.optim.Adadelta(net.parameters(), lr=lr)
#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# Decay LR by a factor of 0.1 every 7 epochs
lr_schedulerr = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
epoch_num = 500

for epoch in range(epoch_num):
    print("------ epoch " + str(epoch) + " -----")
    net.train()

    loss_epoch = 0

    running_loss = 0.0
    running_corrects = 0
    for batch, (trainx, trainy) in enumerate(train_loader):
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
            loss = criterion(y_pred, trainy.squeeze().cuda())
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

    running_loss = 0.0
    running_corrects = 0
    if epoch % 1 == 0:
        net.eval()
        # print("Validating...")
        with torch.no_grad():
            for _, (val_x, val_y) in enumerate(val_loader):
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



