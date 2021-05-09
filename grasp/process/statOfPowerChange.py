'''
collect statistic about power change for each active change across different movement
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

sid=1
sessions=4
movements=4

plot_dir=data_dir + 'PF' + str(sid) +'/ERSD_change/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

movementEpochs=[] # movementEpochs[0] is the epoch of move 1
ch_names=[]
print('Reading all 4 movement epochs.')
for movement in range(movements):
    movementEpochs.append(mne.read_epochs(data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch'+str(movement)+'.fif').pick(picks=activeChannels[sid]))
    ch_names.append(movementEpochs[movement].ch_names)
ch_names=ch_names[0]
ch_names=[str(index)+'-'+name for index,name in zip(activeChannels[6],ch_names)]

fMin,fMax=2,150
fstep=1
freqs=np.arange(fMin,fMax,fstep) #148
n_cycles=freqs

def getIndex(fMin,fMax,fstep,freq):
    freqs=[*range(fMin,fMax,fstep)]
    distance=[abs(fi-freq) for fi in freqs]
    index=distance.index(min(distance))
    return index

decim=4
new_fs=1000/decim
base1=10 #s
base2=13 #s
erds_span=[[1.5,8.0],[1.5,14.0],[1.5,6.0],[1.5,8.0]]
baseline = [int((base1)*new_fs), int((base2)*new_fs)]
erd_change=[] # ers_change[movement][channel][trials....]
ers_change=[]
for movement in range(movements):
    erd_change.append([])
    ers_change.append([])
    for chIndex,chName in enumerate(ch_names):
        erd_change[movement].append([])
        ers_change[movement].append([])
        print('Processing channel '+chName+'.')
        #one_channel=movementEpochs[movement].copy().pick(picks=[chIndex]) # pick the channle below
        one_channel_tf=np.squeeze(tfr_morlet(
            movementEpochs[movement], picks=[chIndex],freqs=freqs, n_cycles=n_cycles,use_fft=True,
            return_itc=False, average=False, decim=decim, n_jobs=1).data)
        # (40, 148, 3751)
        # ERS/ERD of all trials
        for trial in range(40):
            #erd_change[movement][chIndex].append([])
            #ers_change[movement][chIndex].append([])
            base = one_channel_tf[trial,:, baseline[0]:baseline[1]]
            basemean = np.mean(base, 1)
            basestd = np.std(base, 1)
            one_channel_tf[trial] = one_channel_tf[trial] - basemean[:, None]
            one_channel_tf[trial] = one_channel_tf[trial] / basestd[:, None]

            erd0 = getIndex(fMin, fMax, fstep, ERD[0])
            erd1 = getIndex(fMin, fMax, fstep, ERD[1])
            ers0 = getIndex(fMin, fMax, fstep, ERS[0])
            ers1 = getIndex(fMin, fMax, fstep, ERS[1])

            erd = np.mean(one_channel_tf[trial][erd0:erd1, :], 0)
            ers = np.mean(one_channel_tf[trial][ers0:ers1, :], 0)
            change_span=erds_span[movement] # [1.5,8.0]
            erd_change_span=erd[int(change_span[0]*new_fs):int(change_span[1]*new_fs)]
            #erd_change[movement][trial] = max(erd_change_span) - min(erd_change_span)
            erd_change[movement][chIndex].append(max(erd_change_span) - min(erd_change_span))
            ers_change_span = ers[int(change_span[0] * new_fs):int(change_span[1] * new_fs)]
            #ers_change[movement][trial] = max(ers_change_span) - min(ers_change_span)
            ers_change[movement][chIndex].append(max(ers_change_span) - min(ers_change_span))

#ers/d_change[movement][channel][trials....]
fig, ax = plt.subplots()
for channel in range(len(ch_names)):
    ax.clear()
    dataset0=erd_change[0][channel]
    dataset1=erd_change[1][channel]
    dataset2=erd_change[2][channel]
    dataset3=erd_change[3][channel]
    datasets=[dataset0,dataset1,dataset2,dataset3]

    x = np.array([1, 2, 3, 4]) #
    y = [np.mean(dataset) for dataset in datasets]
    e = [np.std(dataset) for dataset in datasets]
    ax.errorbar(x, y, e, linestyle='None', fmt='-o')
    plt.show()
    # save
    figname = plot_dir + 'ERSD_change' + str(channel) + '.png'
    fig.savefig(figname, dpi=400)
    plt.pause(2)



'''
#check 
fig, ax = plt.subplots()

vmin=-3
vmax=5
ax.clear()
im=ax.imshow(one_channel_tf[trial],origin='lower',cmap='RdBu_r',vmin=vmin, vmax=vmax)
ax.set_aspect('auto')
'''
