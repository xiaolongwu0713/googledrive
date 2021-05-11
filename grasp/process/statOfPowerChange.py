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

sid=6
sessions=4
movements=4

plot_dir=data_dir + 'PF' + str(sid) +'/ERSD_stat_change/'
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

crop1=0
crop2=15
decim=4
new_fs=1000/decim
base1=10 #s
base2=13 #s
erds_span=[[1.5,8.0],[1.5,14.0],[1.5,6.0],[1.5,8.0]]
baseline=[[] for _ in range(movements)]
baseline[0] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[1] = [int((14-crop1)*new_fs), int((15-crop1)*new_fs)]
baseline[2] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[3] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]

erd_change=[] # ers_change[movement][channel][trials....]
ers_change=[]
for chIndex,chName in enumerate(ch_names):
    print('Processing channel ' + chName + '.')
    erd_change.append([])
    ers_change.append([])
    for movement in range(movements):
        erd_change[chIndex].append([])
        ers_change[chIndex].append([])
        #one_channel=movementEpochs[movement].copy().pick(picks=[chIndex]) # pick the channle below
        one_channel_tf=np.squeeze(tfr_morlet(
            movementEpochs[movement], picks=[chIndex],freqs=freqs, n_cycles=n_cycles,use_fft=True,
            return_itc=False, average=False, decim=decim, n_jobs=1).data)
        # (40, 148, 3751)
        # ERS/ERD of all trials
        for trial in range(40):
            #erd_change[movement][chIndex].append([])
            #ers_change[movement][chIndex].append([])
            base = one_channel_tf[trial,:, baseline[movement][0]:baseline[movement][1]]
            basemean = np.mean(base, 1)
            basestd = np.std(base, 1)
            one_channel_tf[trial] = one_channel_tf[trial] - basemean[:, None]
            one_channel_tf[trial] = one_channel_tf[trial] / basestd[:, None]

            erd0 = getIndex(fMin, fMax, fstep, ERD[0])
            erd1 = getIndex(fMin, fMax, fstep, ERD[1])
            ers0 = getIndex(fMin, fMax, fstep, ERS[0])
            ers1 = getIndex(fMin, fMax, fstep, ERS[1])

            compare_with=np.mean(one_channel_tf[trial][:,baseline[movement][0]:baseline[movement][1]])
            erd = np.mean(one_channel_tf[trial][erd0:erd1, :], 0)
            ers = np.mean(one_channel_tf[trial][ers0:ers1, :], 0)
            change_span=erds_span[movement] # [1.5,8.0]
            erd_change_span = erd[int(change_span[0] * new_fs):int(change_span[1] * new_fs)]
            #erd_change[movement][chIndex].append((max(erd_change_span) - min(erd_change_span))/max(erd_change_span))
            erd_change[chIndex][movement].append((min(erd_change_span)-compare_with))# / compare_with): RuntimeWarning: divide by zero encountered...
            ers_change_span = ers[int(change_span[0] * new_fs):int(change_span[1] * new_fs)]
            #ers_change[movement][chIndex].append((max(ers_change_span) - min(ers_change_span))/min(ers_change_span))
            ers_change[chIndex][movement].append((max(ers_change_span) - compare_with))# / compare_with)

#ers/d_change[movement][channel][trials....]
fig, ax = plt.subplots()
print('Plotting...')
for channel in range(len(ch_names)):
    ax.clear()
    dataset0=erd_change[channel][0]
    dataset1=erd_change[channel][1]
    dataset2=erd_change[channel][2]
    dataset3=erd_change[channel][3]
    datasets=[dataset0,dataset1,dataset2,dataset3]

    x = np.array([1, 2, 3, 4]) #
    xlabel=['20% MVC slow','26% MVC slow','20% MVC fast','60% MVC fast',]
    y = [np.mean(dataset) for dataset in datasets]
    e = [np.std(dataset) for dataset in datasets]
    ax.errorbar(x, y, e, linestyle='None', fmt='-o')
    ax.set_xticks(x)
    fontdict={'fontsize':8}
    ax.set_xticklabels(xlabel,fontdict=fontdict)
    ax.set_ylabel('Change %')
    #plt.show()
    # save
    figname = plot_dir + 'ERD_stat_change' + str(channel) + '.png'
    fig.savefig(figname, dpi=400)
    plt.pause(0.2)



'''
#check 
fig, ax = plt.subplots()

vmin=-3
vmax=5
ax.clear()
im=ax.imshow(one_channel_tf[trial],origin='lower',cmap='RdBu_r',vmin=vmin, vmax=vmax)
ax.set_aspect('auto')
'''
#x=np.arange(1,10)
#y=x
#plt.plot(x,y)
#ax=plt.gca()
#ax.set_ylabel('Change %')