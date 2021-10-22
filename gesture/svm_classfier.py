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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from common_dl import set_random_seeds
from common_dl import myDataset
from comm_utils import slide_epochs
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

loadPath = project_dir+'preprocessing2.mat'
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
epochs = mne.Epochs(raw, events1, tmin=-3, tmax=4,baseline=None)
epochs=epochs.load_data().pick(picks=activeChan)
epoch_info=epochs.info
# or epoch from 0s to 4s which only contain movement data.
# epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

epoch1=epochs['0'] # 20 trials. 8001 time points per trial for 8s.
epoch2=epochs['1']
epoch3=epochs['2']
epoch4=epochs['3']
epoch5=epochs['4']

epoch1_bsl=epoch1.load_data().copy().crop(-3,0)
epoch2_bsl=epoch1.load_data().copy().crop(-3,0)
epoch3_bsl=epoch1.load_data().copy().crop(-3,0)
epoch4_bsl=epoch1.load_data().copy().crop(-3,0)
epoch5_bsl=epoch1.load_data().copy().crop(-3,0)
list_of_epochs_bsl=[epoch1_bsl,epoch2_bsl,epoch3_bsl,epoch4_bsl,epoch5_bsl]

epoch1=epoch1.load_data().crop(0,4).get_data()
epoch2=epoch2.load_data().crop(0,4).get_data()
epoch3=epoch3.load_data().crop(0,4).get_data()
epoch4=epoch4.load_data().crop(0,4).get_data()
epoch5=epoch5.load_data().crop(0,4).get_data()
list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]
wind=500
stride=250
for i in range(5):
    clas=i + 1
    Xi,y=slide_epochs(list_of_epochs[i],clas,wind, stride)
    tmp_epoch=mne.EpochsArray(Xi,epoch_info)
    events=tmp_epoch.events
    events[:,2]=clas
    tmp_epoch.events=events
    list_of_epochs[i]=tmp_epoch
trialN=Xi.shape[0]

list_of_epochs_bsl_psd=[]
for i in range(5):
    (tmp,freq)=mne.time_frequency.psd_welch(list_of_epochs_bsl[i],average='mean')
    tmp=np.mean(tmp,axis=0) # (208, 129)
    list_of_epochs_bsl_psd.append(tmp)


list_of_epochs_psd=[]
for i in range(5):
    (tmp,freq)=mne.time_frequency.psd_welch(list_of_epochs[i],average='mean') # tmp: (300, 208, 129)
    #tmp=np.mean(tmp,axis=0) # (208, 129)
    list_of_epochs_psd.append(tmp)

# average across below frequency ranges
fbands = [[1,4],[4,8],[8,13],[13,30],[60,75],[75,95],[105,125],[125,145],[155,195]]
chnN=list_of_epochs_psd[0].shape[1]
fbandsN=len(fbands)

list_of_epochs_bsl_psd_avg=[]
for i in range(5):
    tmp_epoch_bsl_psd_avg = np.zeros([chnN, fbandsN])
    for k,fi in enumerate(fbands):
        lowf=fi[0]
        highf=fi[1]
        lowf_index,_=min(enumerate(freq), key=lambda x: abs(x[1] - lowf))
        highf_index,_ = min(enumerate(freq), key=lambda x: abs(x[1] - highf))
        tmp_epoch_bsl_psd_avg[:,k]=np.mean(list_of_epochs_bsl_psd[i][:,lowf_index:highf_index],axis=1)
    list_of_epochs_bsl_psd_avg.append(tmp_epoch_bsl_psd_avg)


list_of_epochs_psd_avg=[]
for i in range(5):
    tmp_epoch_psd_avg = np.zeros([trialN,chnN, fbandsN])
    #tmp_epoch_psd_avg = np.zeros([trialN, chnN*fbandsN])
    for k,fi in enumerate(fbands):
        lowf=fi[0]
        highf=fi[1]
        lowf_index,_=min(enumerate(freq), key=lambda x: abs(x[1] - lowf))
        highf_index,_ = min(enumerate(freq), key=lambda x: abs(x[1] - highf))
        tmp_epoch_psd_avg[:,:,k]=np.mean(list_of_epochs_psd[i][:,:,lowf_index:highf_index],axis=2)
    tmp_epoch_psd_avg=tmp_epoch_psd_avg-list_of_epochs_bsl_psd_avg[i]
    tmp=np.reshape(tmp_epoch_psd_avg,(trialN, chnN*fbandsN))
    list_of_epochs_psd_avg.append(tmp)

list_of_epochs_psd_avg=np.concatenate(list_of_epochs_psd_avg,axis=0) # (1500, 23*9=207)


list_of_labes=[]
for i in range(5):
    trialN=tmp.shape[0]
    label=[[i,]*trialN]
    list_of_labes.append(label)
list_of_labes=np.squeeze(np.asarray(list_of_labes))
list_of_labes=np.squeeze(list_of_labes.reshape((1,-1))) # (1500,)


sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
sss.get_n_splits(list_of_epochs_psd_avg, list_of_labes)

pipe_svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
c_range=np.arange(0.00005,0.0002,0.00001)
g_range=np.arange(0.0005,0.002,0.0001)
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_range,
               'svc__gamma': g_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=8,
                  n_jobs=-1)

gs = gs.fit(xtrain, np.ravel(ytrain))
print(gs.best_score_)
print(gs.best_params_)
clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(xtest, ytest))

accuracy=[]
for train_index, test_index in sss.split(list_of_epochs_psd_avg, list_of_labes):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = list_of_epochs_psd_avg[train_index], list_of_epochs_psd_avg[test_index]
    y_train, y_test = list_of_labes[train_index], list_of_labes[test_index]
    # clf = make_pipeline(SVC(gamma='auto'))
    # input format: X:(samples, feature_number) y:(samples,)
    pipe_svc.fit(X_train, y_train)
    y_predict = pipe_svc.predict(X_test)
    confusion_matrix(y_test, y_predict)
    accu=sum(y_test == y_predict) / len(y_test)
    accuracy.append(accu)
    print("Accuracy: %.2f" % (accu))
accuracy=np.asarray(accuracy)
filename=project_dir+'accuracy_sid'+str(sid)
np.save(filename,accuracy)





