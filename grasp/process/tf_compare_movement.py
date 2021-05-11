'''
Compare TF plot among different movement.
'''
import sys

from grasp.process.channel_settings import activeChannels

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from mne.time_frequency import tfr_morlet, tfr_multitaper
import matplotlib.pyplot as plt
from grasp.config import *


sid=6

plot_dir=data_dir + 'PF' + str(sid) +'/tf_compare_movement/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

sessions=4
movements=4
# vertical lines indicate trigger onset
movementsLines=[[0,2,5,7.5,15],[0,2,11,13.5,15],[0,2,3,5.5,15],[0,2,5,7.5,15]]

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
n_cycles=freqs
#lowCycles=30
#n_cycles=[8]*lowCycles + [50]*(fNum-lowCycles)

averagePower=[]
decim=4
new_fs=1000/decim
for chIndex,chName in enumerate(ch_names):
    averagePower.append([])
    print('Computing TF on ' + str(chIndex) + '/'+str(len(ch_names))+' channel.')
    for movement in range(movements):
        singleMovementEpoch=movementEpochs[movement]
        # decim will decrease the sfreq, so 15s will becomes 5s afterward.
        averagePower[chIndex].append(np.squeeze(tfr_morlet(singleMovementEpoch, picks=[chIndex],
                   freqs=freqs, n_cycles=n_cycles,use_fft=True,return_itc=False, average=True, decim=decim, n_jobs=1).data))

# plot to test the cycle parameter
#channel=0
#averagePower[channel].plot(baseline=(13,14.5), vmin=-4,vmax=4,mode='zscore', title=ch_names[channel]+'_'+str(channel),axes=ax)

# crop the original power data because there is artifact at the beginning and end of the trial.
power=[]
crop1=0
crop2=15
for channel in range(len(ch_names)):
    power.append([])
    for movement in range(movements):
        power[channel].append(averagePower[channel][movement][:,int(crop1*new_fs):int(crop2*new_fs)+1])

'''
# use my own zscore function to plot because there is no different between mine and MNE.
# plot tf for all active channels
for channel in range(len(ch_names)):
    fig, ax = plt.subplots(4,1)
    for movement in range(movements):
        power[channel][movement].plot(baseline=(13,14.5), vmin=-4,vmax=4,mode='zscore', title=ch_names[channel]+'_'+str(channel),axes=ax)
        for x_value in movementLines:
            plt.axvline(x=x_value)
            plt.xticks(movementLines)
        figname=plot_dir+str(channel)+'.png'
        fig.savefig(figname, dpi=400)
        fig.clear()
        ax.clear()
        plt.close(fig)
'''

# return the index for the demanded freq
def getIndex(fMin,fMax,fstep,freq):
    freqs=[*range(fMin,fMax,fstep)]
    distance=[abs(fi-freq) for fi in freqs]
    index=distance.index(min(distance))
    return index

duration=power[0][0].shape[1] # duration 3500=14s*250
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
#np.save('/tmp/power.npy',power)
#power=np.load('/tmp/power.npy')
vmin=-4
vmax=4

# different movement should have different baseline
baseline=[[] for _ in range(movements)]
baseline[0] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[1] = [int((14-crop1)*new_fs), int((15-crop1)*new_fs)]
baseline[2] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]
baseline[3] = [int((10-crop1)*new_fs), int((13-crop1)*new_fs)]


#(300, 5001)
fig, ax = plt.subplots(nrows=4,ncols=1)
print('Ploting out to '+plot_dir+'.')
for channel in range(len(ch_names)):
    for movement in range(movements):
        base=power[channel][movement][:,baseline[movement][0]:baseline[movement][1]]
        basemean=np.mean(base,1)
        basestd=np.std(base,1)
        power[channel][movement]=power[channel][movement]-basemean[:,None]
        power[channel][movement]=power[channel][movement]/basestd[:,None]
        im=ax[movement].imshow(power[channel][movement],origin='lower',cmap='RdBu_r',vmin=vmin, vmax=vmax)
        ax[movement].set_aspect('auto')

        ax[movement].set_xticks(tickAndVertical[movement])
        xlabels = [str(i/new_fs) for i in tickAndVertical[movement]]
        ax[movement].set_xticklabels(xlabels)

        #plot vertical lines
        for x_value in tickAndVertical[movement]:
            ax[movement].axvline(x=(x_value-int(crop1*new_fs)))
        #fig.colorbar(im, ax=ax0)
        fig.colorbar(im, orientation="horizontal",fraction=0.046, pad=0.0,ax=ax[movement])

    # save
    figname = plot_dir + 'tf_compare_movement_'+str(channel) + '.png'
    fig.savefig(figname, dpi=400)

    # clean up plotting area
    for movement in range(movements):
        ax[movement].images[-1].colorbar.remove()
        ax[movement].cla()
plt.close(fig)


