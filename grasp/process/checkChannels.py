import hdf5storage
import mne
import numpy as np
import matplotlib.pyplot as plt

from grasp.config import *
from grasp.process.utils import getRawData, get_trigger

#fittness between ERS/ERD and force level

sid=6
sessions=4 # 4 sessions
movements=4 # 4 movements
session=0 # evaluate the channle on one session

print("Read raw data from disk.")
seegfile=data_raw+'PF'+str(sid)+'/SEEG_Data/PF6_F_'+str(session+1)+'.mat'
mat = hdf5storage.loadmat(seegfile)
raw = mat['Data']
triggerRaw = mat['Data'][29, :]
fs = int(mat['Fs'][0][0])
chnRaw = mat['ChnName']
channels = np.asarray([chnRaw[i][0][0][0] for i in range(len(chnRaw))])  # list with len=126
#ch_names = [channelsName.strip() for channelsName in channels]
ch_names=[str(chi) for chi in [*range(len(channels))]]
ch_types=['seeg'] * len(ch_names)
info = mne.create_info(ch_names=list(ch_names), ch_types=list(ch_types), sfreq=fs)
raw = mne.io.RawArray(raw, info)

# use plot to check data and pick useChannels and triggerChannel
rawd=raw.copy().resample(100)
rawd.copy().plot(scalings='auto')
#useChannels[6]=np.concatenate((np.arange(0,15),np.arange(16,29),np.arange(37,119)))
# find out the trigger channel: 29-36. Pick 29
























