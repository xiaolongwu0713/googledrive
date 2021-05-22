'''
Not freq band info which only extract that power amplitude of that band.
Like the computing of TF-plot using morlet wavelet, here tf will be z-normalized. Then concatenate with raw data.
'''
import sys
import time

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from mne.time_frequency import tfr_morlet, tfr_multitaper
import numpy as np
import mne
import matplotlib.pyplot as plt
from grasp.config import *
from grasp.process.channel_settings import *

sid=6
sessions=4
movements=4

output_dir=data_dir + 'PF' + str(sid) +'/data/'
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

movementEpochs=[] # movementEpochs[0] is the epoch of move 1
ch_names=[]
print('Reading all 4 movement epochs.')
for movement in range(movements):
    movementEpochs.append(mne.read_epochs(data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch'+str(movement)+'.fif').pick(picks=activeChannels[sid]))
    ch_names.append(movementEpochs[movement].ch_names)
ch_names=ch_names[0]
ch_names=[str(index)+'-'+name for index,name in zip(activeChannels[sid],ch_names)]


decim=4
new_fs=int(1000/decim)
raw_data=[]
pick_channels=activeChannels[sid]+[-3,-2,-1]
for movement in range(movements):
    raw_data.append([])
    tmp=mne.read_epochs(data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch' + str(movement) + '.fif').pick(picks=pick_channels)
    raw_data[movement]=tmp.resample(new_fs).get_data()


fMin,fMax=2,125
fstep=1
freqs=np.arange(fMin,fMax,fstep) #148
n_cycles=freqs

def getIndex(fMin,fMax,fstep,freq):
    freqs=[*range(fMin,fMax,fstep)]
    distance=[abs(fi-freq) for fi in freqs]
    index=distance.index(min(distance))
    return index

crop1=0
crop2=15


base1=10 #s
base2=13 #s
erds_span=[[1.5,8.0],[1.5,14.0],[1.5,6.0],[1.5,8.0]]
baseline=[[] for _ in range(movements)]
baseline[0] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[1] = [int((14-crop1)*new_fs), int((15-crop1)*new_fs)]
baseline[2] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[3] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]

band_index=[1,2,3,4]
fband_index=[]
fband_index.append([getIndex(fMin, fMax, fstep, fbands[band_index[0]][0]), getIndex(fMin, fMax, fstep, fbands[band_index[0]][1])])
fband_index.append([getIndex(fMin, fMax, fstep, fbands[band_index[1]][0]), getIndex(fMin, fMax, fstep, fbands[band_index[1]][1])])
fband_index.append([getIndex(fMin, fMax, fstep, fbands[band_index[2]][0]), getIndex(fMin, fMax, fstep, fbands[band_index[2]][1])])
fband_index.append([getIndex(fMin, fMax, fstep, fbands[band_index[3]][0]), getIndex(fMin, fMax, fstep, fbands[band_index[3]][1])])
band_num=len(fband_index)

extra_chn_names_prefix=[str(fbands[i][0])+'-'+str(fbands[i][1]) for i in band_index]

bandEpoch=[]
for movement in range(movements):
    print('Processing movement ' + str(movement) + '.')
    bandEpoch.append([])
    chn_3D=[]
    extra_chn_names=[]
    for chIndex, chName in enumerate(ch_names):
        extra_chn_names.append([prefix+chName for prefix in extra_chn_names_prefix])
        one_channel_tf=np.squeeze(tfr_morlet(
            movementEpochs[movement], picks=[chIndex],freqs=freqs, n_cycles=n_cycles,use_fft=True,
            return_itc=False, average=False, decim=decim, n_jobs=1).data)
        # (40, 148, 3751)(trials, frequencies,times), delete last one to keep align with raw data
        one_channel_tf=one_channel_tf[:,:,:-1] #-->(40, 148, 3750)

        # extract band power
        #erd_change[movement][chIndex].append([])
        #ers_change[movement][chIndex].append([])
        base = one_channel_tf[:,:, baseline[movement][0]:baseline[movement][1]]
        basemean = np.mean(base, 2)
        basestd = np.std(base, 2)
        one_channel_tf = one_channel_tf - basemean[:,:, None]
        one_channel_tf = one_channel_tf / basestd[:,:, None]
        compare_with = np.mean(one_channel_tf[:,:,baseline[movement][0]:baseline[movement][1]],2)
        one_channel_tf=one_channel_tf-compare_with[:,:,None]

        sub_band_3D=[]
        for band in range(band_num):
            sub_band_3D.append([])
            sub_band_3D[band]=np.mean(one_channel_tf[:,fband_index[band][0]:fband_index[band][1],:],1)
        band_3D=np.array(sub_band_3D).transpose(1,0,2) #--> (40, 4, 3751)
        chn_3D.append(band_3D)
    # stack
    bandEpoch[movement]=np.concatenate(chn_3D,1)
# frequency features
extra_chn_names=np.concatenate(extra_chn_names)
ch_names_all=np.append(extra_chn_names,ch_names) #,['emg','emg','stim'])
ch_names_all=np.append(ch_names_all,['emg','emg','stim'])

ch_types=np.concatenate((np.repeat(np.array('seeg'),(len(fband_index)+1)*len(activeChannels[sid])),
                         np.repeat(np.array('emg'),2),np.repeat(np.array('stim'),1)))

# stack raw and frequency data
results=[]
for movement in range(movements):
    results.append([])
    results[movement]=np.concatenate((bandEpoch[movement],raw_data[movement]),1)
    print(results[movement].shape)
    info = mne.create_info(ch_names=list(ch_names_all), ch_types=list(ch_types), sfreq=new_fs)
    epoch=mne.EpochsArray(results[movement],info)
    epoch.save(output_dir + 'move_normalized_band_epoch'+str(movement)+'.fif', overwrite=True)



