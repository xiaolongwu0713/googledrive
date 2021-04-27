from mne.time_frequency import tfr_morlet, tfr_multitaper
import numpy as np
import mne
import matplotlib.pyplot as plt
from grasp.config import *

# Epoch the data before doing this.

sid=2
plot_dir=data_raw + 'PF' + str(sid) +'/tfPlot/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

sessions=4
movements=4
vminmax=4
# vertical lines indicate trigger onset
movementsLines=[[0,2,5,7.5,15],[0,2,11,13.5,15],[0,2,3,5.5,15],[0,2,5,7.5,15]]

movementEpochs=[] # movementEpochs[0] is the epoch of move 1
#print('Reading all 4 movement epochs.')
#for movement in range(movements):
#    movementEpochs.append(mne.read_epochs(data_raw + 'PF' + str(sid) + '/data/' + 'move'+str(movement)+'epoch.fif').pick(picks=['seeg']))

## just read one will be OK, save memory
chooseOneMovement=0
movementEpochs.append(mne.read_epochs(data_raw + 'PF' + str(sid) + '/data/' + 'move'+str(chooseOneMovement)+'epoch.fif').pick(picks=['seeg']))
ch_names=movementEpochs[0].ch_names


# Choose one epoch to evaluate channe activity.
oneEpoch=chooseOneMovement
movementLines=movementsLines[oneEpoch]

# pick some channel doing some testing or pick all channel during process.
#pickSubChannels=[4,5,6,7,8,9]
pickSubChannels=list(range(len(ch_names)))
ch_names=[ch_names[i] for i in pickSubChannels]
singleMovementEpoch=movementEpochs[oneEpoch].pick(picks=pickSubChannels)

## frequency analysis
# define frequencies of interest (log-spaced)
fMin,fMax=2,150
fNum=fMin-fMax
fstep=1
decim=3
new_fs=1000/decim
cycleMin,cycleMax=1,150
cycleNum=300
#freqs = np.linspace(fMin,fMax, num=fNum)
freqs=np.arange(fMin,fMax,fstep)
n_cycles = freqs #np.linspace(cycleMin,cycleMax, num=cycleNum)  # different number of cycle per frequency
baseline=[13,14.5]
averagePower=[]
for chIndex,chName in enumerate(ch_names):
    averagePower.append([])
    # decim will decrease the sfreq, so 15s will becomes 5s afterward.
    averagePower[chIndex]=tfr_morlet(singleMovementEpoch, picks=[chIndex],
               freqs=freqs, n_cycles=n_cycles,use_fft=True,return_itc=False, average=True, decim=decim, n_jobs=1)

'''
# use my one zscore function to plot because there is no different between mine and MNE.
# plot tf for all channels
for channel in range(len(ch_names)):
    fig, ax = plt.subplots()
    averagePower[channel].plot(baseline=(13,14.5), vmin=-4,vmax=4,mode='zscore', title=ch_names[channel]+'_'+str(channel),axes=ax)
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

# convert MNE to numpy and
# crop the original power data because there is artifect at the begining and end of the trial.
power=[]
crop1=0.5
crop2=14.5
for channel in range(len(ch_names)):
    power.append([])
    # Crop here will cause the movement line mis-aligned.
    power[channel]=np.squeeze(averagePower[channel].data)
    # crop to zeros.
    power[channel][:,:int(crop1*new_fs)]=0
    power[channel][:,int(crop2*new_fs):]=0

#np.save('/tmp/power.npy',power)
#power=np.load('/tmp/power.npy')
vmin=-4
vmax=4
baseline = [int(13*new_fs), int(14.5*new_fs)]
#(300, 5001)
fig, ax = plt.subplots(nrows=2,ncols=1, sharex=True)
#fig, ax = plt.subplots(2)
ax0=ax[0]
ax1=ax[1]
for channel in range(len(ch_names)):
    base=power[channel][:,baseline[0]:baseline[1]]
    basemean=np.mean(base,1)
    basestd=np.std(base,1)
    power[channel]=power[channel]-basemean[:,None]
    power[channel]=power[channel]/basestd[:,None]
    im=ax0.imshow(power[channel],origin='lower',cmap='RdBu_r',vmin=vmin, vmax=vmax)
    ax0.set_aspect('auto')

    # xtick shoule include common points and vertical line points.
    # change common points from[0,1000,2000,..5000] to [0,3,6,9,12,15]
    tickAndVertical=list(range(0,6000,1000))
    # include vertical line points
    for i in movementLines:
        tickAndVertical.append(int(i*new_fs))
    # get unique value
    tmp=list(set(tickAndVertical))
    tmp.sort()
    tickAndVertical=tmp
    ax0.set_xticks(tickAndVertical)

    # set ticklabels
    x = [int(i/new_fs) for i in np.arange(0, 6000, 1000)]
    x=list(set(x+movementLines))
    ax0.set_xticklabels(x)

    #plot vertical lines
    for x_value in movementLines:
        ax0.axvline(x=x_value*new_fs)
    #fig.colorbar(im, ax=ax0)
    fig.colorbar(im, orientation="horizontal",fraction=0.046, pad=0.2,ax=ax0)

    # ERD/DRS plot
    erd0 = getIndex(fMin, fMax, fstep, ERD[0])
    erd1 = getIndex(fMin, fMax, fstep, ERD[1])
    ers0 = getIndex(fMin, fMax, fstep, ERS[0])
    ers1 = getIndex(fMin, fMax, fstep, ERS[1])

    erd=np.mean(power[channel][erd0:erd1,int(crop1*new_fs):int(crop2*new_fs)],0)
    ers=np.mean(power[channel][ers0:ers1,int(crop1*new_fs):int(crop2*new_fs)],0)

    # plot the ERS/ERD
    ax1.plot(erd)
    ax1.plot(ers)
    # save
    figname = plot_dir + 'TFandERSD_'+str(channel) + '.png'
    fig.savefig(figname, dpi=400)

    # clean up plotting area
    ax0.images[-1].colorbar.remove()
    ax0.cla()
    ax1.cla()
plt.close(fig)


