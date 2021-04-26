import hdf5storage
import mne
import numpy as np
import matplotlib.pyplot as plt

from grasp.config import *
from channel_settings import *
from grasp.process.utils import getRawData, get_trigger

'''
Function: pick up the non-signal channel and trigger channel by plotting and visual check.
'''
sid=2
sessions=4 # 4 sessions
movements=4 # 4 movements
session=0 # evaluate the channle on one session

print("Read raw data from disk.")
seegfile=data_raw+'PF'+str(sid)+'/SEEG_Data/PF'+str(sid)+'_F_'+str(session+1)+'.mat'
mat = hdf5storage.loadmat(seegfile)
raw = mat['Data']
#triggerRaw = mat['Data'][29, :]
fs = int(mat['Fs'][0][0])
chnRaw = mat['ChnName']
channels = np.asarray([chnRaw[i][0][0][0] for i in range(len(chnRaw))])  # list with len=126
#ch_names = [channelsName.strip() for channelsName in channels]
ch_index_str=[str(chi) for chi in [*range(len(channels))]]
ch_index=[*range(len(channels))] #147
ch_types=['seeg'] * len(ch_index_str)
info = mne.create_info(ch_names=list(ch_index_str), ch_types=list(ch_types), sfreq=fs)
raw = mne.io.RawArray(raw, info)

# use plot to check data and pick useChannels and triggerChannel
rawd=raw.copy().resample(100)
del mat, raw
rawd.plot(scalings='auto',n_channels=3,duration=30.0,start=50.0)
#useChannels[6]=np.co---------ncatenate((np.arange(0,15),np-------.arange(16,29),np.arange(37,119)))
# find out the trigger channel: 29-36. Pick 29
badChannels=[14,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,145,146] # 19
channelTemp1=[item for item in ch_index if item not in badChannels] #147-1=128
# exam again
rawd.copy().pick(picks=channelTemp1).plot(scalings='auto',n_channels=10,duration=200.0,start=0.0)
# exam the non-signal channel---
rawd.copy().pick(picks=badChannels).plot(scalings='auto',n_channels=3,duration=100.0,start=0.0)
# 37-42 is trigger(use 38), exam the other
rawd.copy().pick(picks=badChannels2).plot(scalings='auto',n_channels=3,duration=100.0,start=0.0)
badChannels2=[item for item in badChannels if item !=38] #147-1=128
















