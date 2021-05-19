'''
Compare TF plot of different movement.
'''
import sys

from grasp.process.channel_settings import activeChannels

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from mne.time_frequency import tfr_morlet, tfr_multitaper
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from grasp.config import *


sid=6

plot_dir=data_dir + 'PF' + str(sid) +'/choose_bands/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

sessions=4
movements=4
# vertical lines indicate trigger onset
movementsLines=[[0,2,5,7.5,15],[0,2,11,13.5,15],[0,2,3,5.5,15],[0,2,5,7.5,15]]

# return the index for the demanded freq
def getIndex(fMin,fMax,fstep,freq):
    freqs=[*range(fMin,fMax,fstep)]
    distance=[abs(fi-freq) for fi in freqs]
    index=distance.index(min(distance))
    return index


movementEpochs=[] # movementEpochs[0] is the epoch of move 1
print('Reading all 4 movement epochs.')
for movement in range(movements):
    movementEpochs.append(mne.read_epochs(
        data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch'+str(movement)+'.fif').pick(picks=activeChannels[sid]))
ch_names=movementEpochs[0].ch_names

## frequency analysis
# define frequencies of interest (log-spaced)
fMin,fMax=2,150
fstep=1
freqs=np.arange(fMin,fMax,fstep) #148
fNum=freqs.shape[0]
#freqs = np.linspace(fMin,fMax, num=fNum)
cycleMin,cycleMax=8,50
cycleNum=fNum
#n_cycles = np.linspace(cycleMin,cycleMax, num=cycleNum)  # different number of cycle per frequency
groups=5
rates=[2,2.5,3,4,5]
num_per_group=int(fNum/groups)
n_cycles=[]
for g in range(groups):
    if g < groups -1:
        tmp=[int(i) for i in freqs[g*num_per_group:(g+1)*num_per_group]/rates[g]]
    elif g==groups -1:
        tmp = [int(i) for i in freqs[g * num_per_group:] / rates[g]]
    n_cycles.extend(tmp)

decim=4
new_fs=1000/decim

vmin=-4
vmax=4

crop1=0
crop2=15
colors=[]
# different movement should have different baseline
baseline=[[] for _ in range(movements)]
baseline[0] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[1] = [int((14-crop1)*new_fs), int((15-crop1)*new_fs)]
baseline[2] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[3] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]


erds_change = []
ch_power_avg = []
fig,ax=plt.subplots()
for chIndex,chName in enumerate(ch_names):
    erds_change.append([])
    ch_power_avg.append([])
    print('Computing TF on ' + str(chIndex) + '/'+str(len(ch_names))+' channel.')
    for movement in range(movements):
        if movement==3:
            baseline=(14,14.8)
        else:
            boaseline=(10,13)
        erds_change[chIndex].append([])
        ch_power_avg[chIndex].append([])
        singleMovementEpoch=movementEpochs[movement]
        # decim will decrease the sfreq, so 15s will becomes 5s afterward.
        tmp = tfr_morlet(singleMovementEpoch, picks=[chIndex],
                                         freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True,
                                         decim=decim, n_jobs=1)
        tmp.plot(baseline=baseline, vmin=-4, vmax=4, mode='zscore',axes=ax)
        filename=plot_dir+str(chIndex)+'move'+str(movement)+'.png'
        fig.savefig(filename, dpi=400)
        img = plt.gca().images
        img[-1].colorbar.remove()
        ax.clear()
        #fig.clf()

