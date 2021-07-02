'''
Choose frequency window ers_wind=5 which maxmize the difference between task and baseline as ers frequency range.
So ers will only consider frequency range from ers_wind[0] to ers_wind[1].
Same as erd_wind.

When slide the window, erd will start from begin frequence to erd_end_f=30 HZ, and ers will start from ers_start_f=50Hz till end frq.

Reason: ERS very week compare to ERD. ERS will vanish when mean across large freq range because there are relative short large ERD between ERS.
'''
import sys

from grasp.process.channel_settings import activeChannels
from grasp.process.signalProcessUtils import getIndex

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from mne.time_frequency import tfr_morlet, tfr_multitaper
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from grasp.config import *


sid=10
# fast testing
activeChannels[sid]=activeChannels[sid][4]

plot_dir=data_dir + 'PF' + str(sid) +'/power_change_2bands/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

sessions=4
movements=4
decim=4
new_fs=1000/decim

# vertical lines indicate trigger onset
movementsLines=[[0,2,5,7.5,15],[0,2,11,13.5,15],[0,2,3,5.5,15],[0,2,5,7.5,15]]
task_durations=[]
for i in range(len(movementsLines)):
    task_durations.append([])
    task_durations[i]=[int(new_fs*movementsLines[i][1]),int(new_fs*movementsLines[i][3])]

movementEpochs=[] # movementEpochs[0] is the epoch of move 1
print('Reading all 4 movement epochs.')
for movement in range(movements):
    movementEpochs.append(mne.read_epochs(
        data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch'+str(movement)+'.fif').pick(picks=activeChannels[sid]))
ch_names=movementEpochs[0].ch_names

## frequency analysis
# define frequencies of interest (log-spaced)
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


vmin=-4
vmax=4

# no crop here, will eliminate edge artifact later.
crop1=0
crop2=15
colors=[]
# different movement should have different baseline
baseline=[[] for _ in range(movements)]
baseline[0] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[1] = [int((14-crop1)*new_fs), int((15-crop1)*new_fs)-1]
baseline[2] = [int((13-crop1)*new_fs), int((14.5-crop1)*new_fs)]
baseline[3] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]

sub_bands_number=2
erd_wind=10
ers_wind=2
erd_end_f=30
ers_start_f=30
erds_change = []
ch_power_avg = []

for chIndex,chName in enumerate(ch_names):
    erds_change.append([])
    ch_power_avg.append([])
    print('Computing TF on ' + str(chIndex) + '/'+str(len(ch_names))+' channel.')
    for movement in range(movements):
        erds_change[chIndex].append([])
        ch_power_avg[chIndex].append([])
        singleMovementEpoch=movementEpochs[movement]
        # decim will decrease the sfreq, so 15s will becomes 5s afterward.
        tmp = tfr_morlet(singleMovementEpoch, picks=[chIndex],
                                         freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True,
                                         decim=decim, n_jobs=1)
        ch_power_avg[chIndex][movement] = np.squeeze(tmp.data)
        # The very big artifact at begainning and end of the TF will cause ERS vanish after normalization.
        ch_power=np.squeeze(tfr_morlet(singleMovementEpoch, picks=[chIndex],
                   freqs=freqs, n_cycles=n_cycles,use_fft=True,return_itc=False, average=False, decim=decim, n_jobs=1).data)
        # eliminate the artifact at the begining and end of the tf matrix
        edge=20
        ch_power[:, :, :edge] = ch_power[:, :, edge][:,:,None]
        ch_power[:, :, -edge:] = ch_power[:, :, -edge][:,:,None]
        base_avg = ch_power_avg[chIndex][movement][:, baseline[movement][0]:baseline[movement][1]]
        base_avg_mean = np.mean(base_avg, 1)
        base_avg_std = np.std(base_avg, 1)

        base = ch_power[:,:, baseline[movement][0]:baseline[movement][1]]
        basemean = np.mean(base, 2)
        basestd= np.std(base, 2)

        #Method: z-score
        #ch_power = ch_power-basemean[:, :, np.newaxis]
        #ch_power = ch_power/basestd[:, :, np.newaxis]
        # ch_power_avg[chIndex][movement] = ch_power_avg[chIndex][movement] - base_avg_mean[:, None]
        # ch_power_avg[chIndex][movement] = ch_power_avg[chIndex][movement] / base_avg_std[:, None]

        #Method: db is better
        ch_power_avg[chIndex][movement] = 10 * np.log10(ch_power_avg[chIndex][movement] / base_avg_mean[:, None])
        ch_power_norm = 10*np.log10(ch_power / basemean[:, :, None])

        #mean across task duration
        mean1=np.mean(ch_power_norm[:,:,task_durations[movement][0]:task_durations[movement][1]], axis=2)
        mean2=np.mean(mean1,axis=0)

        erd_start_f_index=getIndex(fMin, fMax, fstep, fMin)
        erd_end_f_index = getIndex(fMin, fMax, fstep, erd_end_f) # 30

        ers_start_f_index = getIndex(fMin, fMax, fstep, ers_start_f) # 50
        ers_end_f_index = getIndex(fMin, fMax, fstep, fMax)

        # moving window average
        erd_wind_avg = np.convolve(mean2[erd_start_f_index:erd_end_f_index], np.ones(erd_wind) / erd_wind, mode='valid')
        ers_wind_avg = np.convolve(mean2[ers_start_f_index:ers_end_f_index], np.ones(ers_wind) / ers_wind,mode='valid')
        # find the max active frequency
        erd_f1=erd_wind_avg.argmin(axis=0)
        erd_f2=erd_f1 + erd_wind
        ers_f1 = ers_start_f_index+ers_wind_avg.argmax(axis=0)
        ers_f2 = ers_start_f_index+ers_f1 + ers_wind

        #Todo: why ERS all lie below 0?
        move_up=2
        erd = np.mean(ch_power_norm[:, erd_f1:erd_f2, :], axis=1)+move_up
        ers = np.mean(ch_power_norm[:, ers_f1:ers_f2, :], axis=1)+move_up
        #compare_with[band] = np.mean(np.mean(ch_power_norm[:,index1:index2,:],axis=1)[:,baseline[movement][0]:baseline[movement][1]],axis=1)
        #devide by compare_with[band][:,np.newaxis] will cause result very unstable.
        erds_change[chIndex][movement].append(erd)# - compare_with[band][:,np.newaxis] # result is better if not subtract compare_with.
        erds_change[chIndex][movement].append(ers)

duration=ch_power_avg[0][0].shape[1] # duration 3500=14s*250
tickAndVertical=[]
for movement in range(movements):
    tickAndVertical.append([0,duration-1]) # first and last points don't need to change

# include vertical line points
for movement in range(movements):
    tmp=tickAndVertical[movement]
    for i in movementsLines[movement][1:-1]: # 0s and 15s excluded
        tmp.append(int((i-crop1)*new_fs))
    tmp=list(set(tmp))
    tmp.sort()
    tickAndVertical[movement]=tmp

print('Plot out to '+ plot_dir+ '.')
clrs = sns.color_palette("husl", 5)
fig = plt.figure(figsize=(10, 8))

#choose_bands_to_plot=range(len(fbands))
name_your_band=['beta','gamma']
#fig, ax=plt.subplots(2,2,sharex=True,sharey=True, squeeze=True)
for channel in range(len(ch_names)):
    if channel == int(len(ch_names)/2):
        print('Half way through.')
    outer = gridspec.GridSpec(2, 2, wspace=0.02, hspace=0.02)
    for movement in range(movements):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[movement], wspace=0.1, hspace=0.1)
        # plot 2D TF
        ax1 = plt.Subplot(fig, inner[0])
        im = ax1.imshow(ch_power_avg[channel][movement], origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax1.set_aspect('auto')
        fig.add_subplot(ax1)
        fig.colorbar(im, orientation="horizontal", fraction=0.046, pad=0.02, ax=ax1)

        ax1.set_ylabel('Frequency')
        ax1.set_xticks(tickAndVertical[movement])
        xlabels = [str(i / new_fs)+'s' for i in tickAndVertical[movement]]
        ax1.set_xticklabels(xlabels,rotation=0, ha='right',fontsize=5)
        ax1.tick_params(axis='x', which='major', pad=0.2)

        for x_value in tickAndVertical[movement]:
            ax1.axvline(x=(x_value - int(crop1 * new_fs)))

        # plot ERDS
        ax2 = plt.Subplot(fig, inner[1])
        fig.add_subplot(ax2)
        #with sns.axes_style("darkgrid"):
        for band in np.arange(sub_bands_number):
            mean = np.mean(erds_change[channel][movement][band],axis=0)
            times=np.arange(len(mean))
            sdt = np.std(erds_change[channel][movement][band], axis=0)
            ax2.plot(times,mean)
            ax2.fill_between(times,mean-sdt, mean+sdt ,alpha=0.3, facecolor=clrs[band])
        ax2.legend(name_your_band)
        ax2.set_ylabel('Change %')
        ax2.set_xticks(tickAndVertical[movement])
        xlabels = [str(i / new_fs) + 's' for i in tickAndVertical[movement]]
        ax2.set_xticklabels(xlabels, rotation=0, ha='right', fontsize=5)
        ax2.tick_params(axis='x', which='major', pad=0.2)

        if movement%2!=0:
            ax1.axes.yaxis.set_visible(False)
            ax2.axes.yaxis.set_visible(False)

    filename = plot_dir + str(channel) + '.png'
    fig.savefig(filename, dpi=400)
    fig.clear()
