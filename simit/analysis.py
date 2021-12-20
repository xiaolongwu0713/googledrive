import hdf5storage
import matplotlib.pyplot as plt
import scipy.io
import os
import mne
import numpy as np
data_dir='/Volumes/Samsung_T5/data/simit/XY20210823/CCEP_1/preprocessing/'
result_dir='/Users/long/Documents/data/work/simit/'
filename=data_dir+'raw_EEG.mat'
fs=4000
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
#mat2=hdf5storage.loadmat(filename)

mat=scipy.io.loadmat(filename)
raw=mat['EEG'] # raw is a np structured arrays: https://numpy.org/doc/stable/user/basics.rec.html
del mat

data=raw['data'][0][0]
data_subtract=np.random.rand(data.shape[0]-1,data.shape[1])
for chi in range(data_subtract.shape[0]):
    data_subtract[chi,:]=data[chi+1,:]-data[chi,:]

events=raw['event'][0][0]
#type=events['type']
latency_tmp=events['latency'][0]
latency=[tmp[0][0] for tmp in latency_tmp]
events=[]
for lati in latency:
    eventi=[lati,0,1]
    events.append(eventi)
events=np.asarray(events)

del raw
durations=[latency[i]-latency[i-1] for i in range(1,len(latency))]

chn_num=data.shape[0]
# stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
chn_names=["seeg"]*chn_num
chn_types=["seeg"]*chn_num
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
info_sub = mne.create_info(ch_names=list(chn_names)[:-1], ch_types=list(chn_types)[:-1], sfreq=fs)
raw = mne.io.RawArray(data, info)
raw_sub=mne.io.RawArray(data_subtract, info_sub)
epochs = mne.Epochs(raw, events, tmin=-1, tmax=4,baseline=None)
raw.copy().pick(picks=[24,25]).plot(events=events,scalings=dict(seeg=500000))
raw_sub.copy().pick(picks=[24,25]).plot(duration=20.0, start=40.0,scalings=dict(seeg=500000))

raw_sub.copy().pick(picks=[24,25]).filter(l_freq=15,h_freq=200).plot(duration=20.0, start=40.0,scalings=dict(seeg=500000))
lf=raw_sub.copy().filter(l_freq=5,h_freq=200) # high-pass filter
lf.copy().plot(events=events,scalings=dict(seeg=500000))
raw_sub.plot_psd(fmax=200)
