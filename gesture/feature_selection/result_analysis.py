import hdf5storage
import numpy as np
from gesture.config import *
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sid=10
fs=1000
result_dir=data_dir+'training_result/selection'+'/P'+str(sid)+'/' + 'selection/gumbel/'
#result_dir='/Users/long/OneDrive/share/selection/gumbel/3/P10'
scores = np.load(result_dir + 'epoch_scores.npy') # (train acc, val acc)
h=np.load(result_dir+'HH.npy')
s=np.load(result_dir+'SS.npy') # selection
z=np.load(result_dir+'ZZ.npy') # probability
h=np.squeeze(h)
z=np.squeeze(z)

mean_entropy=np.mean(h,axis=1)
plt.plot(scores[:,0])
plt.plot(scores[:,1])
plt.plot(mean_entropy)

# best training epoch: how to find the best epoch: lowest entropy + highest val acc
#best_train= np.where(scores == max(scores[:,1]))
best_epoch=70
# plot matrix
plt.imshow(z[-1,130:170,:])

selected_channels=np.argmax(z[-1,:,:],axis=0)
selected_channels=list(set(selected_channels)) # [143, 144, 146, 147, 148, 149, 150, 151, 152, 167]

loadPath = data_dir+'preprocessing'+'/P'+str(sid)+'/preprocessing2.mat'
mat=hdf5storage.loadmat(loadPath)
data = mat['Datacell']
channelNum=int(mat['channelNum'][0,0])
data=np.concatenate((data[0,0],data[0,1]),0)
del mat

# stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
chn_names=np.append(["seeg"]*channelNum,["emg0","emg1","stim_trigger","stim_emg"])
chn_types=np.append(["seeg"]*channelNum,["emg","emg","stim","stim"])
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
raw = mne.io.RawArray(data.transpose(), info)

# gesture/events type: 1,2,3,4,5
events0 = mne.find_events(raw, stim_channel='stim_trigger')
events1 = mne.find_events(raw, stim_channel='stim_emg')
# events number should start from 0: 0,1,2,3,4, instead of 1,2,3,4,5
events0=events0-[0,0,1]
events1=events1-[0,0,1]

#print(events[:5])  # show the first 5
# Epoch from 4s before(idle) until 4s after(movement) stim1.
raw=raw.pick(["seeg"])
rawd=raw.load_data().get_data() # (208, 1052092)
raw_selected=raw.pick(selected_channels).get_data() # (10, 1052092)

rawd[143,:10]











