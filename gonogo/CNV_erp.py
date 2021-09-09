# choose channels from tf analysis, then plot the CNV ERP

import hdf5storage
import os
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet

from gonogo.config import *

sid=5 #4
data_dir='/Volumes/Samsung_T5/data/ruijin/gonogo/preprocessing/P'+str(sid)
plot_dir=data_dir + '/tfPlot/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

#Session_num,UseChn,EmgChn,TrigChn = get_channel_setting(sid)
#original_fs=[Frequencies[i,1] for i in range(Frequencies.shape[0]) if Frequencies[i,0] == pn][0]
loadPath = data_dir+'/preprocessing/preprocessingv2.mat'
mat=hdf5storage.loadmat(loadPath)
fs=mat['Fs']
rtime=mat['ReactionTime']
rtime=np.concatenate((rtime[0,0],rtime[0,1]),axis=0)
data=mat['DATA']
data=np.concatenate((data[0,0],data[0,1]),axis=0) #(2160440, 63)
events=mat['Trigger']
events=np.concatenate((events[0,0],events[0,1]),axis=0) # two sessions
events[:, [1,2]] = events[:, [2,1]] # swap 1st and 2nd column to: timepoint, duration, event code
events=events.astype(int)

del mat

chn_num=data.shape[1]

event1=events[(events[:,2]==1)]
event2=events[(events[:,2]==2)]
event34_index=[i or j for (i,j) in zip((events[:,2]==3), (events[:,2]==4))]
event34=events[event34_index]
event56_index=[i or j for (i,j) in zip((events[:,2]==5), (events[:,2]==6))]
event56=events[event56_index]
event1112_index=[i or j for (i,j) in zip((events[:,2]==11), (events[:,2]==12))]
event1112=events[event1112_index]
event2122_index=[i or j for (i,j) in zip((events[:,2]==21), (events[:,2]==22))]
event2122=events[event2122_index]
event11=events[(events[:,2]==11)]
#event12=events[(events[:,2]==12)]

chn_names=np.asarray(["seeg"]*chn_num)
chn_types=np.asarray(["seeg"]*chn_num)
info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
raw = mne.io.RawArray(data.transpose(), info)

epoch1=mne.Epochs(raw, event1, tmin=-1, tmax=4,baseline=None) # fixed 3s
epoch2=mne.Epochs(raw, event2, tmin=-1, tmax=4,baseline=None) # fixed 3s
epoch34=mne.Epochs(raw, event34, tmin=-3, tmax=1,baseline=None) # varying reaction time, 3s maximum
epoch56=mne.Epochs(raw, event56, tmin=-3, tmax=1,baseline=None) # varying reaction time, 3s maximum
epoch1112=mne.Epochs(raw, event1112, tmin=-7, tmax=6,baseline=None) # 3s task cue and 3s executing cue
epoch2122=mne.Epochs(raw, event2122, tmin=0, tmax=3,baseline=None) # 3s executing cue
epoch11=mne.Epochs(raw, event11, tmin=-7, tmax=3.5,baseline=None)
epoch11a=mne.Epochs(raw, event11, tmin=-7, tmax=4.0,baseline=None)
#epoch12=mne.Epochs(raw, event12, tmin=-7, tmax=6,baseline=None)

cnv1112 = epoch1112.load_data().copy().pick(picks=['seeg']).filter(l_freq=0.1, h_freq=1) # evoked data
cnv11_avg=cnv1112['11'].average(method='mean') # evoked.data returns the underlying data
cnv12_avg=cnv1112['12'].average(method='mean')
cnv11_avg.plot()

cnv1112b = epoch1112.load_data().copy().pick(picks=['seeg']).filter(l_freq=0.1, h_freq=35) # evoked data
cnv11b_avg=cnv1112b['11'].average(method='mean') # evoked.data returns the underlying data
cnv12b_avg=cnv1112b['12'].average(method='mean')
cnv11b_avg.pick(picks=[46]).plot()


cnv11 = epoch11.load_data().copy().pick(picks=['seeg']).filter(l_freq=0.05, h_freq=2)
cnv11_avg=cnv11.average(method='mean')
cnv11_avg.pick(picks=[42]).plot()

cnv11a = epoch11a.load_data().copy().pick(picks=['seeg']).filter(l_freq=0.05, h_freq=2)
cnv11a_avg=cnv11a.average(method='mean')
cnv11a_avg.pick(picks=[46]).plot()