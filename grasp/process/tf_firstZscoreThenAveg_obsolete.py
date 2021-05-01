from mne.time_frequency import tfr_morlet, tfr_multitaper
import numpy as np
import mne
import matplotlib.pyplot as plt
from grasp.config import *

plot_dir=root_dir+'grasp/process/timeFreq2/'

sid=6
sessions=4
movements=4
vminmax=4
# vertical lines indicate trigger onset
movementsLines=[[0,2,5,7.5,15],[0,2,11,13.5,15],[0,2,3,5.5,15],[0,2,5,7.5,15]]

movementEpochs=[] # movementEpochs[0] is the epoch of move 1
print('Reading all 4 movement epochs.')
for movement in range(movements):
    movementEpochs.append(mne.read_epochs(data_raw + 'PF' + str(sid) + '/data/' + 'move'+str(movement)+'epoch.fif').pick(picks=['seeg']))
ch_names=movementEpochs[0].ch_names


# Choose one epoch to evaluate channe activity.
oneEpoch=0
movementLines=movementsLines[oneEpoch]
pickSubChannels=[4,5,6,7,8,9]
ch_names=[ch_names[i] for i in pickSubChannels]
singleMovementEpoch=movementEpochs[oneEpoch].pick(picks=pickSubChannels)

## frequency analysis
# define frequencies of interest (log-spaced)
fMin,fMax=2,150
fNum=300
cycleMin,cycleMax=1,150
cycleNum=300
freqs = np.linspace(fMin,fMax, num=fNum)
n_cycles = np.linspace(cycleMin,cycleMax, num=cycleNum)  # different number of cycle per frequency
baseline=[13,14.5]
averagePower=[]
for chIndex,chName in enumerate(ch_names):
    averagePower.append([])
    tmpPower=tfr_morlet(singleMovementEpoch, picks=[chIndex],
               freqs=freqs, n_cycles=n_cycles,use_fft=True,return_itc=False, average=False, decim=3, n_jobs=1)
    tmpPower.apply_baseline(baseline, mode="percent")
    averagePower[chIndex] = tmpPower
# plot tf for all channels
for channel in range(len(ch_names)):
    fig, ax = plt.subplots()
    averagePower[channel].average().plot(baseline=(13,14.5), vmin=-1,vmax=1, title=ch_names[channel]+'_'+str(channel),axes=ax)
    for x_value in movementLines:
        plt.axvline(x=x_value)
        plt.xticks(movementLines)
    figname=plot_dir+str(channel)+'.png'
    fig.savefig(figname, dpi=400)
    fig.clear()
    ax.clear()
    plt.close(fig)
