'''
collect statistic about power change for each active change across different movement
'''
import sys
import time

from grasp.process.signalProcessUtils import getIndex

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from mne.time_frequency import tfr_morlet, tfr_multitaper
import numpy as np
import mne
import matplotlib.pyplot as plt
from grasp.config import *
from grasp.process.channel_settings import *



sid = 2
movements=4
# fast testing
#activeChannels[sid]=activeChannels[sid][2]

plot_dir = data_dir + 'PF' + str(sid) + '/ERSD_stat_change/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

movementEpochs = []  # movementEpochs[0] is the epoch of move 1
ch_names = []
print('Reading all 4 movement epochs.')
for movement in range(movements):
    movementEpochs.append(
        mne.read_epochs(data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch' + str(movement) + '.fif').pick(
            picks=activeChannels[sid]))
    ch_names.append(movementEpochs[movement].ch_names)
ch_names = ch_names[0]
ch_names = [str(index) + '-' + name for index, name in zip(activeChannels[6], ch_names)]

n_cycles_mthod='stage' # or: equal
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
if n_cycles_mthod=='equal':
    n_cycles=freqs
elif n_cycles_mthod=='stage':
    n_cycles=[]
    for g in range(groups):
        if g < groups -1:
            tmp=[int(i) for i in freqs[g*num_per_group:(g+1)*num_per_group]/rates[g]]
        elif g==groups -1:
            tmp = [int(i) for i in freqs[g * num_per_group:] / rates[g]]
        n_cycles.extend(tmp)


crop1 = 0
crop2 = 15
decim = 4
new_fs = 1000 / decim
base1 = 10  # s
base2 = 13  # s
erds_span = [[1.5, 8.0], [1.5, 14.0], [1.5, 6.0], [1.5, 8.0]]
baseline = [[] for _ in range(movements)]
baseline[0] = [int((10 - crop1) * new_fs), int((13 - crop1) * new_fs)]
baseline[1] = [int((14 - crop1) * new_fs), int((15 - crop1) * new_fs)]
baseline[2] = [int((10 - crop1) * new_fs), int((13 - crop1) * new_fs)]
baseline[3] = [int((10 - crop1) * new_fs), int((13 - crop1) * new_fs)]

movementsLines=[[0,2,5,7.5,15],[0,2,11,13.5,15],[0,2,3,5.5,15],[0,2,5,7.5,15]]
task_durations=[]
for i in range(len(movementsLines)):
    task_durations.append([])
    task_durations[i]=[int(new_fs*movementsLines[i][1]),int(new_fs*movementsLines[i][3])]


sub_bands_number = 2
erd_wind = 10
ers_wind = 2
erd_end_f = 30
ers_start_f = 30
normalization = 'z-socre'  # 'z-socre'/'db'

erd_change = []  # ers_change[movement][channel][trials....]
ers_change = []
for chIndex, chName in enumerate(ch_names):
    print('Computing TF on ' + str(chIndex) + '/' + str(len(ch_names)) + ' channel.')
    # print('Processing channel ' + chName + '.')
    erd_change.append([])
    ers_change.append([])
    for movement in range(movements):
        erd_change[chIndex].append([])
        ers_change[chIndex].append([])
        # one_channel=movementEpochs[movement].copy().pick(picks=[chIndex]) # pick the channle below
        one_channel_tf = np.squeeze(tfr_morlet(
            movementEpochs[movement], picks=[chIndex], freqs=freqs, n_cycles=n_cycles, use_fft=True,
            return_itc=False, average=False, decim=decim, n_jobs=1).data)
        # (40, 148, 3751)
        # ERS/ERD of all trials
        for trial in range(40):
            # erd_change[movement][chIndex].append([])
            # ers_change[movement][chIndex].append([])
            base = one_channel_tf[trial, :, baseline[movement][0]:baseline[movement][1]]
            basemean = np.mean(base, 1)
            basestd = np.std(base, 1)

            if normalization == 'z-score':
                # Method:z-score
                one_channel_tf[trial] = one_channel_tf[trial] - basemean[:, None]
                one_channel_tf[trial] = one_channel_tf[trial] / basestd[:, None]
            elif normalization == 'db':
                # Method:db
                one_channel_tf[trial] = 10 * np.log10(one_channel_tf[trial] / basemean[:, None])

            mean_change = np.mean(one_channel_tf[trial][:, task_durations[movement][0]:task_durations[movement][1]], axis=1)

            erd_start_f_index = getIndex(fMin, fMax, fstep, fMin)
            erd_end_f_index = getIndex(fMin, fMax, fstep, erd_end_f)  # 30

            ers_start_f_index = getIndex(fMin, fMax, fstep, ers_start_f)  # 50
            ers_end_f_index = getIndex(fMin, fMax, fstep, fMax)

            # moving window average
            erd_wind_avg = np.convolve(mean_change[erd_start_f_index:erd_end_f_index], np.ones(erd_wind) / erd_wind,
                                       mode='valid')
            ers_wind_avg = np.convolve(mean_change[ers_start_f_index:ers_end_f_index], np.ones(ers_wind) / ers_wind,
                                       mode='valid')
            # find the max active frequency
            erd_f1 = erd_wind_avg.argmin(axis=0)
            erd_f2 = erd_f1 + erd_wind
            ers_f1 = ers_start_f_index + ers_wind_avg.argmax(axis=0)
            ers_f2 = ers_start_f_index + ers_f1 + ers_wind

            erd_change[chIndex][movement].append(np.mean(one_channel_tf[erd_f1:erd_f2,task_durations[movement][0]:task_durations[movement][1]]))
            ers_change[chIndex][movement].append(np.mean(one_channel_tf[ers_f1:ers_f2,task_durations[movement][0]:task_durations[movement][1]]))


# ers/d_change[movement][channel][trials....]
fig, ax = plt.subplots()
print('Plotting...')
for channel in range(len(ch_names)):
    erd_move0 = erd_change[channel][0]
    erd_move1 = erd_change[channel][1]
    erd_move2 = erd_change[channel][2]
    erd_move3 = erd_change[channel][3]

    ers_move0 = ers_change[channel][0]
    ers_move1 = ers_change[channel][1]
    ers_move2 = ers_change[channel][2]
    ers_move3 = ers_change[channel][3]

    erd_datasets = [erd_move0, erd_move1, erd_move2, erd_move3]
    ers_datasets = [ers_move0, ers_move1, ers_move2, ers_move3]

    x = np.array([1, 2, 3, 4])  #
    xlabel = ['20% MVC slow', '26% MVC slow', '20% MVC fast', '60% MVC fast', ]
    y_erd = [np.mean(dataset) for dataset in erd_datasets]
    y_ers = [np.mean(dataset) for dataset in ers_datasets]
    e_erd = [np.std(dataset) for dataset in erd_datasets]
    e_ers = [np.std(dataset) for dataset in ers_datasets]

    ax.errorbar(x=x, y=y_erd, yerr=e_erd, fmt='-o', ecolor='orange', elinewidth=1, ms=5, mfc='wheat', mec='salmon',capsize=3)
    ax.errorbar(x=x, y=y_ers, yerr=e_ers, fmt='-o', ecolor='blue', elinewidth=1, ms=5, mfc='wheat', mec='salmon', capsize=3)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.legend(['ERD', 'ERS'])
    ax.set_xticks(x)
    fontdict = {'fontsize': 8}
    ax.set_xticklabels(xlabel, fontdict=fontdict)
    ax.set_ylabel('Change %')
    # plt.show()
    # save
    figname = plot_dir + 'ERSD_stat_change' + str(channel) + '.png'
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
# x=np.arange(1,10)
# y=x
# plt.plot(x,y)
# ax=plt.gca()
# ax.set_ylabel('Change %')