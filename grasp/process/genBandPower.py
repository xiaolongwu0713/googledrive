import hdf5storage
import mne
import numpy as np
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell
from grasp.config import *
from grasp.process.utils import get_trigger, getMovement, get_trigger_normal, getForceData, \
    genSubTargetForce, getRawData
import matplotlib.pyplot as plt


sid=6
plot_dir=data_raw + 'PF' + str(sid) +'/process/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Note: input: session in [0,1,2,3]; channels=activeChannels or useChannels
channels=activeChannels
#activeChannels.append(stim) # 29 is stim(trigger) channel
#activeChannels.sort()

movements=4
movementEpochs=[] # movementEpochs[0] is the epoch of move 1
for movement in range(movements):
    movementEpochs.append(mne.read_epochs(data_raw + 'PF' + str(sid) + '/data/' + 'move'+str(movement)+'epoch.fif'))

# psd before band pass
print("Plot psd after bandpass.")
fig,ax=plt.subplots()
movementEpochs[0].plot_psd(fmax=300,picks=['seeg'],ax=ax,xscale='linear',average=True,spatial_colors=False)
figname=root_dir+plot_dir+'psdOfEpochBeforeBandpass'
fig.savefig(figname)
plt.close(fig)

bandEpochs=[] # bandEpochs[0]=[deltaEpoch,thetaEpoch,....]
for movement in range(movements):
    print('Bandpass epoch '+str(movement))
    bandEpochs.append([])
    deltaEpoch = movementEpochs[movement].copy().pick(picks=['seeg']).filter(l_freq=fbands[0][0], h_freq=fbands[0][1])  # (19, 648081)
    thetaEpoch = movementEpochs[movement].copy().pick(['seeg']).filter(l_freq=fbands[1][0], h_freq=fbands[1][1])  # ..
    alphaEpoch = movementEpochs[movement].copy().pick(picks=['seeg']).filter(l_freq=fbands[2][0], h_freq=fbands[2][1])  # ..
    betaEpoch = movementEpochs[movement].copy().pick(picks=['seeg']).filter(l_freq=fbands[3][0], h_freq=fbands[3][1])  # ..
    gammaEpoch = movementEpochs[movement].copy().pick(picks=['seeg']).filter(l_freq=fbands[4][0], h_freq=fbands[4][1])  # ..
    bandEpochs[movement] = [deltaEpoch, thetaEpoch, alphaEpoch, betaEpoch, gammaEpoch]
# psd after band pass
print("Plot psd after bandpass.")
movement=0 # Take first movement as an example and plot its 5 bands PSD
fig,ax=plt.subplots()
for band in range(len(fbands)):
    bandEpochs[movement][band].copy().plot_psd(fmax=300,picks=['seeg'],ax=ax,xscale='linear',area_mode=None,average=True,spatial_colors=False)
figname=root_dir+plot_dir+'psdOf5BandsOfEpoch0AfterBandpass'
fig.savefig(figname)
plt.close(fig)

# Apply hilbert
# Question: how to use apply_function??
# Todo: calcualte power
#a=alpha.copy().pick(picks=[0,]).apply_hilbert(envelope=True).crop(tmin=0,tmax=0.1)
#np_power_arg=[2]
#b=a.copy().apply_function(np.power,args=np_power_arg)
print("Applying hilbert.")
for movement in range(movements):
    for band in range(len(fbands)):
        bandEpochs[movement][band].apply_hilbert(picks=['seeg'],envelope=True)


# CommonOPS
#raw.save(data_dir+'raw.fif')
#for i in range(len(bands)):
#    bands[i].save(data_dir+bandsName[i]+'.fif')
# CommonOPS
# load data from fif and assign to bands THEN to [delta,theta,alpha,beta,gamma]
#raw=mne.io.read_raw_fif(data_dir+'raw.fif',preload=True)
#bands=[[],[],[],[],[]] # has to be listOflist
#bandsName=['delta','theta','alpha','beta','gamma']
#for i in range(len(bandsName)):
#    filename=data_dir+bandsName[i]+'.fif'
#    bands[i]=mne.io.read_raw_fif(filename)
#[delta,theta,alpha,beta,gamma]=bands
# access data: bands[0],band[1].... or delta, theta ....

# stack all bands epoch into one epoch for each movement
print("Stacking all 5 bands together.")
#bandsName=['delta','theta','alpha','beta','gamma']
bandsName=['d','t','a','b','g']# ValueError: Channel names cannot be longer than 15 characters
for movement in range(movements): # change the ch_name, otherwise no way to concatenate
    for band in range(len(fbands)):
        for c in bandEpochs[movement][band].ch_names:
            bandEpochs[movement][band].rename_channels({c: bandsName[band] + '_' + c})

# stack them together, make sure the 2 force and 1 stim channel are at the bottom
moveAndBandEpochs=[]
bandAllEpochs=[]
for movement in range(movements):
    moveAndBandEpochs.append([])
    bandAllEpochs=bandEpochs[movement][0].add_channels([bandEpochs[movement][1],bandEpochs[movement][2],
                  bandEpochs[movement][3],bandEpochs[movement][4]],force_update_info=True)
    moveAndBandEpochs[movement]=bandAllEpochs.add_channels([movementEpochs[movement]],force_update_info=True)

# sanity check
trials=moveAndBandEpochs[0].get_data(picks=[-1,]) #(40, 1, 15001)
plt.plot(trials[0,0,:])

#Save epochs
print('Saving all 4 epochs.')
for movement in range(movements):
    moveAndBandEpochs[movement].save(data_raw + 'PF' + str(sid) + '/data/' + 'moveAndBandEpoch'+str(movement)+'.fif', overwrite=True)

















