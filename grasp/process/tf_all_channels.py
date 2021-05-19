# tf_firstAvegThenZscore.py
import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from mne.time_frequency import tfr_morlet, tfr_multitaper
import numpy as np
import mne
import matplotlib.pyplot as plt
from grasp.config import *

# Epoch the data before doing this.

sid=6

plot_dir=data_dir + 'PF' + str(sid) +'/tfPlot/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

sessions=4
movements=4
# vertical lines indicate trigger onset
movementsLines=[[0,2,5,7.5,15],[0,2,11,13.5,15],[0,2,3,5.5,15],[0,2,5,7.5,15]]

movementEpochs=[] # movementEpochs[0] is the epoch of move 1
#print('Reading all 4 movement epochs.')
#for movement in range(movements):
#    movementEpochs.append(mne.read_epochs(data_raw + 'PF' + str(sid) + '/data/' + 'move'+str(movement)+'epoch.fif').pick(picks=['seeg']))

## You can change this to evaluate other movement epoch.
chooseOneMovement=0
print('Evaluate on epoch '+str(chooseOneMovement)+'.')
movementEpochs.append(mne.read_epochs(data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch'+str(chooseOneMovement)+'.fif').pick(picks=['seeg']))
ch_names=movementEpochs[0].ch_names


# Choose one epoch to evaluate channe activity.
oneEpoch=chooseOneMovement
movementLines=movementsLines[oneEpoch]

# pick some channel doing some testing or pick all channel during process.
#pickSubChannels=[4,5,6,7,8,9]
pickSubChannels=list(range(len(ch_names)))
print('Evaluate on '+str(int(len(pickSubChannels)))+' channels.')
ch_names=[ch_names[i] for i in pickSubChannels]
singleMovementEpoch=movementEpochs[0].pick(picks=ch_names)

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
    if chIndex%20 == 0:
        print('TF analysis on '+str(chIndex)+'th channel.')
    averagePower.append([])
    # decim will decrease the sfreq, so 15s will becomes 5s afterward.
    averagePower[chIndex]=np.squeeze(tfr_morlet(singleMovementEpoch, picks=[chIndex],
               freqs=freqs, n_cycles=n_cycles,use_fft=True,return_itc=False, average=True, decim=decim, n_jobs=1).data)
# plot to test the cycle parameter
#channel=0
#averagePower[channel].plot(baseline=(13,14.5), vmin=-4,vmax=4,mode='zscore', title=ch_names[channel]+'_'+str(channel),axes=ax)

# crop the original power data because there is artifact at the beginning and end of the trial.
power=[]
crop1=0
crop2=15
for channel in range(len(ch_names)):
    power.append([])
    power[channel]=averagePower[channel][:,int(crop1*new_fs):int(crop2*new_fs)]

'''
# use my own zscore function to plot because there is no different between mine and MNE.
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

duration=power[0].shape[1] # duration 3500=14s*250
tickAndVertical=[0,duration-1] # first and last points don't need to change
# include vertical line points
for i in movementLines[1:-1]: # 0s and 15s excluded
    tickAndVertical.append(int((i-crop1)*new_fs))
tmp=list(set(tickAndVertical))
tmp.sort()
tickAndVertical=tmp
#np.save('/tmp/power.npy',power)
#power=np.load('/tmp/power.npy')
vmin=-4
vmax=4
if chooseOneMovement in [0,2,3]:
    base1=10 #s
    base2=13 #s
elif chooseOneMovement==1:
    base1 = 14  # s
    base2 = 14.5  # s
else:
    raise SystemExit("Choose one epoch: 0,1,2,3.")
baseline = [int((base1-crop1)*new_fs), int((base2-crop1)*new_fs)]

#(300, 5001)
fig, ax = plt.subplots(nrows=2,ncols=1, sharex=True)
#fig, ax = plt.subplots(2)
ax0=ax[0]
ax1=ax[1]
print('Ploting out to '+plot_dir+'.')
for channel in range(len(ch_names)):
    if channel%20 == 0:
        print('Ploting '+str(channel)+'th channel.')
    base=power[channel][:,baseline[0]:baseline[1]]
    basemean=np.mean(base,1)
    basestd=np.std(base,1)
    power[channel]=power[channel]-basemean[:,None]
    power[channel]=power[channel]/basestd[:,None]
    im=ax0.imshow(power[channel],origin='lower',cmap='RdBu_r',vmin=vmin, vmax=vmax)
    ax0.set_aspect('auto')

    ax0.set_xticks(tickAndVertical)
    #x = [crop1,2.0, 5.0, 7.5,crop2]
    x = [crop1, movementLines[1], movementLines[2], movementLines[3], crop2]
    ax0.set_xticklabels(x)

    #plot vertical lines
    for x_value in movementLines[1:-1]:
        ax0.axvline(x=(x_value-crop1)*new_fs)
    #fig.colorbar(im, ax=ax0)
    fig.colorbar(im, orientation="horizontal",fraction=0.046, pad=0.02,ax=ax0)

    # ERD/DRS plot
    erd0 = getIndex(fMin, fMax, fstep, ERD[0])
    erd1 = getIndex(fMin, fMax, fstep, ERD[1])
    ers0 = getIndex(fMin, fMax, fstep, ERS[0])
    ers1 = getIndex(fMin, fMax, fstep, ERS[1])

    erd=np.mean(power[channel][erd0:erd1,:],0)
    ers=np.mean(power[channel][ers0:ers1,:],0)

    # plot the ERS/ERD
    ax1.plot(erd)
    ax1.plot(ers)
    for x_value in movementLines[1:-1]:
        ax1.axvline(x=(x_value-crop1)*new_fs)
    # save
    figname = plot_dir + 'TFandERSD_'+str(channel) + '.png'
    fig.savefig(figname, dpi=400)

    # clean up plotting area
    ax0.images[-1].colorbar.remove()
    ax0.cla()
    ax1.cla()
plt.close(fig)


