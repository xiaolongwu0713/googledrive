'''
collect statistic about power change for each active change across different movement
Todo: ERS/ERD is calculated using max/min amplitude. When try to use mean value for both, ERS is below zero. Don't understand the reason.
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
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

sid = 10
movements=4
# fast testing
#activeChannels[sid]=activeChannels[sid][13:15]


plot_dir=data_dir + 'PF' + str(sid) + '/ERSD_stat_change/'
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

crop1=0
crop2=15
decim=4
new_fs=1000/decim
base1=10 #s
base2=13 #s
#erds_span=[[1.5,8.0],[1.5,14.0],[1.5,6.0],[1.5,8.0]]
baseline=[[] for _ in range(movements)]
baseline[0] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[1] = [int((14-crop1)*new_fs), int((15-crop1)*new_fs)]
baseline[2] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[3] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]

movementsLines=[[0,2,5,7.5,15],[0,2,11,13.5,15],[0,2,3,5.5,15],[0,2,5,7.5,15]]
task_durations=[]
for i in range(len(movementsLines)):
    task_durations.append([])
    task_durations[i]=[int(new_fs*movementsLines[i][1]),int(new_fs*movementsLines[i][3])]


normalization = 'db'  # 'z-socre'/'db'

erd_change=[] # ers_change[movement][channel][trials....]
ers_change=[]
for chIndex,chName in enumerate(ch_names):
    print('Computing TF on ' + str(chIndex) + '/' + str(len(ch_names)) + ' channel.')
    #print('Processing channel ' + chName + '.')
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

            if normalization == 'z-score':
                # Method:z-score
                one_channel_tf[trial] = one_channel_tf[trial] - basemean[:, None]
                one_channel_tf[trial] = one_channel_tf[trial] / basestd[:, None]
            elif normalization == 'db':
                # Method:db
                one_channel_tf[trial] = 10 * np.log10(one_channel_tf[trial] / basemean[:, None])


            erd0 = getIndex(fMin, fMax, fstep, ERD[0])
            erd1 = getIndex(fMin, fMax, fstep, ERD[1])
            ers0 = getIndex(fMin, fMax, fstep, ERS[0])
            ers1 = getIndex(fMin, fMax, fstep, ERS[1])

            erd = np.mean(one_channel_tf[trial][erd0:erd1, :], 0) # It's a line.
            ers = np.mean(one_channel_tf[trial][ers0:ers1, :], 0)
            erd_compare_with = np.mean(erd[baseline[movement][0]:baseline[movement][1]])
            ers_compare_with = np.mean(ers[baseline[movement][0]:baseline[movement][1]])
            #erd_change[movement][chIndex].append((max(erd_change_span) - min(erd_change_span))/max(erd_change_span))
            #erd_change[chIndex][movement].append((min(erd_change_span)-compare_with))# / compare_with): RuntimeWarning: divide by zero encountered...
            erd_change[chIndex][movement].append((np.min(erd[task_durations[movement][0]:task_durations[movement][1]]))-erd_compare_with)
            #ers_change[movement][chIndex].append((max(ers_change_span) - min(ers_change_span))/min(ers_change_span))
            #ers_change[chIndex][movement].append((max(ers_change_span) - compare_with))# / compare_with)
            ers_change[chIndex][movement].append((np.max(ers[task_durations[movement][0]:task_durations[movement][1]]))-ers_compare_with)

# ers/d_change[movement][channel][trials....]
fig, ax = plt.subplots()
#fig.set_rasterized(True)
print('Plotting...')
for channel in range(len(ch_names)):
    ax.clear()
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
    xlabel = ['20% MVC slow', '60% MVC slow', '20% MVC fast', '60% MVC fast']
    y_erd = [np.mean(dataset) for dataset in erd_datasets]
    y_ers = [np.mean(dataset) for dataset in ers_datasets]
    e_erd = [np.std(dataset) for dataset in erd_datasets]
    e_ers = [np.std(dataset) for dataset in ers_datasets]

    #ax.clear()
    ax.errorbar(x=x, y=y_erd, yerr=e_erd, fmt='-o',ecolor='orange',elinewidth=1,ms=5,mfc='wheat',mec='salmon',capsize=3)
    ax.errorbar(x=x, y=y_ers, yerr=e_ers, fmt='-o',ecolor='blue',elinewidth=1,ms=5,mfc='wheat',mec='salmon',capsize=3)


    # place the legend on y=0 horizon line. noted that this line may vary in different plot, so calculate the position first.
    yticks = ax.get_yticks()
    ypos = 0
    for i in range(len(yticks)):
        if yticks[i] == 0:
            ypos = i
    y_min, y_max = ax.get_ylim()
    yticks = [(tick - y_min) / (y_max - y_min) for tick in yticks]

    ax.legend(['ERD', 'ERS'], loc="lower left", bbox_to_anchor=(0.8, yticks[ypos]),fontsize='small')
    ax.axhline(y=0, color='r', linestyle='--')

    ax.set_xticks(x)
    fontdict = {'fontsize': 8}
    ax.set_xticklabels(xlabel, fontdict=fontdict)
    ax.set_ylabel('Change %')
    # plt.show()
    # save
    #ax.set_rasterized(True)
    figname = plot_dir + 'ERSD_stat_change' + str(channel) + '.pdf'
    #fig.savefig(figname, format='eps',dpi=300)
    fig.savefig(figname) #, dpi=400)
    #plt.pause(0.2)


# confidence analysis