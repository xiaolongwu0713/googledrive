'''
best so far: PF10-4
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


sid=10
# fast testing
#activeChannels[10]=activeChannels[10][4]

plot_dir=data_dir + 'PF' + str(sid) +'/power_change/'
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
#n_cycles=freqs
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
baseline[1] = [int((14-crop1)*new_fs), int((15-crop1)*new_fs)-1]
baseline[2] = [int((13-crop1)*new_fs), int((14.5-crop1)*new_fs)]
baseline[3] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]

fbands=[] #delta, theta, alpha,beta,gamma
fbands.append([0.5, 4])
fbands.append([4, 8])
fbands.append([8, 20]) # mu(motor cortex)/alpha(visual cortex)
fbands.append([20, 60])
fbands.append([61, 125])
#choose_bands_to_plot=range(len(fbands))
choose_bands_to_plot=[2,4]

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
        #tmp.plot(baseline=(14, 14.5), vmin=-4, vmax=4, mode='zscore')
        ch_power=np.squeeze(tfr_morlet(singleMovementEpoch, picks=[chIndex],
                   freqs=freqs, n_cycles=n_cycles,use_fft=True,return_itc=False, average=False, decim=decim, n_jobs=1).data)
        # plot
        base_avg = ch_power_avg[chIndex][movement][:, baseline[movement][0]:baseline[movement][1]]
        base_avg_mean = np.mean(base_avg, 1)
        base_avg_std = np.std(base_avg, 1)
        ch_power_avg[chIndex][movement] = ch_power_avg[chIndex][movement] - base_avg_mean[:, None]
        ch_power_avg[chIndex][movement] = ch_power_avg[chIndex][movement] / base_avg_std[:, None]

        base = ch_power[:,:, baseline[movement][0]:baseline[movement][1]]
        basemean = np.mean(base, 2)
        basestd= np.std(base, 2)

        #Method: z-score
        #ch_power = ch_power-basemean[:, :, np.newaxis]
        #ch_power = ch_power/basestd[:, :, np.newaxis]
        #Method: db
        ch_power = 10*np.log10(ch_power / basemean[:, :, np.newaxis])

        f_index=[]
        erds=[]
        compare_with=[]
        for band in range(len(fbands)):
            erds.append([])
            f_index.append([])
            compare_with.append([])
            erds_change[chIndex][movement].append([])
            index1 = getIndex(fMin, fMax, fstep, fbands[band][0])
            index2 = getIndex(fMin, fMax, fstep, fbands[band][1])
            erds[band] = np.mean(ch_power[:,index1:index2,:],axis=1)
            compare_with[band] = np.mean(np.mean(ch_power[:,index1:index2,:],axis=1)[:,baseline[movement][0]:baseline[movement][1]],axis=1)
            #devide by compare_with[band][:,np.newaxis] will cause result very unstable.
            erds_change[chIndex][movement][band] = erds[band]# - compare_with[band][:,np.newaxis] # result is better if not subtract compare_with.


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

        ax1.set_xticks(tickAndVertical[movement])
        xlabels = [str(i / new_fs) for i in tickAndVertical[movement]]
        ax1.set_xticklabels(xlabels)

        # plot vertical lines
        for x_value in tickAndVertical[movement]:
            ax1.axvline(x=(x_value - int(crop1 * new_fs)))

        # plot ERDS
        ax2 = plt.Subplot(fig, inner[1])
        fig.add_subplot(ax2)
        #with sns.axes_style("darkgrid"):
        for band in choose_bands_to_plot:
            mean = np.mean(erds_change[channel][movement][band],axis=0)
            times=np.arange(len(mean))
            sdt = np.std(erds_change[channel][movement][band], axis=0)
            ax2.plot(times,mean)
            ax2.fill_between(times,mean-sdt, mean+sdt ,alpha=0.3, facecolor=clrs[band])
        ax2.legend(['band'+str(i) for i in choose_bands_to_plot])
    filename = plot_dir + str(channel) + '.png'
    fig.savefig(filename, dpi=400)
    fig.clear()








