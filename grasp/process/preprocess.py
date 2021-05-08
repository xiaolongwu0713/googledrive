'''
Process workflow:
1, choose the useChannels, trigger channels and discard any invalide channels using checkChannels.py
2, update useChannels, trigger channels in the grasp.channel_settings.py file
3, run preprocess.py to generate epoch data. One epoch means one movement type.
4, choose activeChannels
'''
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from mne.time_frequency import tfr_morlet
from sklearn import preprocessing

from grasp.process.utils import get_trigger, genSubTargetForce, getRawData, getMovement, getForceData, \
    get_trigger_normal, getRawDataInEdf, getMovement_sid10And16
import numpy as np
import mne
import matplotlib.pyplot as plt
from grasp.config import *
from grasp.process.channel_settings import *

# first subject: sid=6
sid=16
sessions=4
movements=4

#plot_dir=root_dir+'grasp/process/result/'
plot_dir=data_dir + 'PF' + str(sid) +'/process/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# info
# useChannels=[1:15,17:29,38:119]
fs=1000 # downsample to fs=1000

## load channel
seegfiles=[]
triggerfiles=[]
forcefiles=[]
for i in range(sessions):
    if sid==16:
        seegfiles.append(data_dir + 'PF' + str(sid) + '/SEEG_Data/PF' + str(sid) + '_F_' + str(i + 1) + '.edf')
    else:
        seegfiles.append(data_dir + 'PF' + str(sid) + '/SEEG_Data/PF' + str(sid) + '_F_' + str(i + 1) + '.mat')
for i in range(sessions):
    triggerfiles.append(data_dir+'PF'+str(sid)+'/Trigger_Data/Trigger_Information_'+str(i+1)+'.mat')
for i in range(sessions):
    forcefiles.append(data_dir+'PF'+str(sid)+'/Force_Data/'+str(i+1)+'-'+str(i+2)+'.mat')


### 1, raw data
#def getRawData(seegfile,useChannel,triggerChannel,new fs):
triggerRaw=[]
myraw=[]
chn_names=[]
print("Reading raw data.")
# sid 16 raw data in edf format
if sid==16:
    for i in range(sessions):
        rawTmp, triggerRawTmp,chn_namesTmp=getRawDataInEdf(seegfiles[i],useChannels[sid],triggerChannels[sid],fs)
        myraw.append(rawTmp)
        triggerRaw.append(triggerRawTmp)
        chn_names.append(chn_namesTmp)
    chn_names=chn_names[0] # channles will remain the same in all sessions.
    chnNumber=len(chn_names)
else:
    for i in range(sessions):
        rawTmp, triggerRawTmp,chn_namesTmp=getRawData(seegfiles[i],useChannels[sid],triggerChannels[sid],fs)
        myraw.append(rawTmp)
        triggerRaw.append(triggerRawTmp)
        chn_names.append(chn_namesTmp)
    chn_names=chn_names[0] # channles will remain the same in all sessions.
    chnNumber=len(chn_names)
    del rawTmp


### 2, experiment setting
movements=[]
for i in range(sessions):
    if sid==10 or sid==16:
        movements.append(getMovement_sid10And16(triggerfiles[i]))
    else:
        movements.append(getMovement(triggerfiles[i]))
allMovements=np.concatenate(movements)

### 4 format the trigger channel
fig,ax=plt.subplots(2,2)
triggers=[]
for i in range(sessions):
    if sid==6 and i==0: # session 0 of sid=6 need some special treatment.
        tmp=get_trigger(triggerRaw[i])
        triggers.append(tmp)
        tindex = np.nonzero(triggers[i])[0]
        triggers[i][tindex] = movements[i]
    else:
        tmp=get_trigger_normal(sid,triggerRaw[i])
        triggers.append(tmp)
        tindex = np.nonzero(triggers[i])[0]
        triggers[i][tindex] = movements[i]
    trialNum=len(np.nonzero(tmp)[0])
    ax[i//2][i-(i//2)*2].plot(tmp) # // means round down to int.
    ax[i//2][i-(i//2)*2].text(0.8, 0.8, str(trialNum)+' trials.', fontsize=10, transform=ax[i//2][i-(i//2)*2].transAxes)
figname = plot_dir+'trigger_of_all_sessions.png'
fig.savefig(figname,dpi=400)
plt.close(fig)

### 5, force data
# padding force data equal to trigger lenght
forces=[]
fig,ax=plt.subplots(2,2)
for i in range(sessions):
    tmp=getForceData(sid,forcefiles[i], triggers[i],fs)
    # take care of some abnormal force
    if sid==10 and i==0:
        very_unlikely_number=2.012345678901234
        # 2.012345678901234 is clear manually setup, eliminate this trial in later stage
        tmp[110000:150000] = [tmp[i] if 1.26<tmp[i]<2.0 else very_unlikely_number for i in range(110000,150000)]
    forces.append(tmp) # will down sample to fs
    ax[i // 2][i - (i // 2) * 2].plot(tmp)  # // means round down to int.
    ax[i // 2][i - (i // 2) * 2].text(0.3, 0.9, 'Session '+str(i) , fontsize=10,transform=ax[i // 2][i - (i // 2) * 2].transAxes)
    tindex=np.nonzero(triggers[i])[0]
    ax[i // 2][i - (i // 2) * 2].xaxis.set_tick_params(labelsize=5)
    ax[i // 2][i - (i // 2) * 2].set_xticks(tindex)
    ax[i // 2][i - (i // 2) * 2].set_xticklabels([*range(len(tindex))])
figname = plot_dir+'force_of_all_sessions.png'
fig.savefig(figname,dpi=400)
plt.close(fig)

### 6 generate the target force
targetForces=[]
fig,ax=plt.subplots(2,2)
for i in range(sessions):
    # create target force data equal to trigger lenght
    tmp=np.ones((triggers[i].shape[0]))*0.05 # 1 D.
    indexAll=np.nonzero(triggers[i])[0] # 40
    for (index,j) in zip(indexAll,np.arange(indexAll.shape[0])):
        tmp[index:index+15000]=genSubTargetForce(movements[i][j],fs)
    targetForces.append(tmp)  # (648081,)
    ax[i // 2][i - (i // 2) * 2].plot(tmp)  # // means round down to int.
    ax[i // 2][i - (i // 2) * 2].text(0.3, 0.9, 'Session ' + str(i), fontsize=10,transform=ax[i // 2][i - (i // 2) * 2].transAxes)
figname = plot_dir +'targetForce_of_all_sessions.png'
fig.savefig(figname, dpi=400)
plt.close(fig)

print("Plotting of force, target and trigger.")
for i in range(sessions):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 2))
    forces[i] = np.squeeze(scaler.fit_transform(forces[i][:,None]))
    targetForces[i] = np.squeeze(scaler.fit_transform(targetForces[i][:,None]))
    tindex=np.nonzero(triggers[i])[0]
    fig, ax = plt.subplots()
    plt.ion()
    ax.clear()
    ax2 = ax.twiny()

    ax.plot(triggers[i], label='Trigger', linewidth=0.1) # (648081,)
    ax.plot(np.squeeze(forces[i]), label='Force',linewidth=0.1) # (1, 648081)
    ax.plot(np.squeeze(targetForces[i]), label='TargetForce',linewidth=0.1) #(648081,)
    # mark the movement type
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tindex)
    ax2.set_xticklabels(movements[i])
    # index trial
    ax.xaxis.set_tick_params(labelsize=5)
    ax.set_xticks(tindex)
    ax.set_xticklabels([*range(len(tindex))])
    #plt.legend()
    figname = plot_dir+'triggerAndForceAndTargetS' + str(i) + '.png'
    fig.savefig(figname,dpi=400)
    plt.close(fig)


## 7 concatenate all 4 session
print("Concatenate raw data from all 4 sessions.")
rawOf4=np.concatenate((myraw[0],myraw[1],myraw[2],myraw[3]),axis=1) #(110, 2742378)
del myraw
forceOf4=np.concatenate((forces[0],forces[1],forces[2],forces[3])) #(1, 2742378)
targetForceOf4=np.concatenate((targetForces[0],targetForces[1],targetForces[2],targetForces[3])) #(2742378,)
triggerOf4=np.concatenate((triggers[0],triggers[1],triggers[2],triggers[3])) #(2742378,)


print("Plotting of force for all 4 session.")
fig, ax = plt.subplots()
plt.ion()
ax.clear()
ax.plot(triggerOf4, label='Trigger', linewidth=0.01) # (648081,)
ax.plot(forceOf4, label='Force',linewidth=0.01) # (1, 648081)
ax.plot(targetForceOf4, label='TargetForce',linewidth=0.01) #(648081,)
plt.legend()
figname = plot_dir+'ALLtriggerAndForceAndTarget.pdf'
fig.savefig(figname)
plt.close(fig)


### 7 concatenat: seeg data + real force + target force + trigger
myraw=np.concatenate((rawOf4,forceOf4[None,:], targetForceOf4[None,:],triggerOf4[None,:]),axis=0) #(113, 648081)
del rawOf4


### create info and raw data
ch_names=np.append(chn_names,['force','target','stim']) # events, emg. total 112 channels = 110+2
#ch_types=np.repeat(np.array('seeg'),126)
ch_types=np.concatenate((np.repeat(np.array('seeg'),chnNumber),np.repeat(np.array('emg'),2),np.repeat(np.array('stim'),1)))
info = mne.create_info(ch_names=list(ch_names), ch_types=list(ch_types), sfreq=fs)
raw = mne.io.RawArray(myraw, info)

### events
# TODO: how to load events from file
#myevents=loadData(31,1,'events')
#event_dict = {'move1': 1, 'move2': 2, 'move3': 3,'move4': 4, 'move5': 5}
# nme can find event from channels of raw data
events = mne.find_events(raw, stim_channel='stim',consecutive=False)
allMovementt=events[:,2]
if (allMovementt==allMovements).all():
    print("All movements sorted corrected.")

# NOTE: some common raw data operation
# DONE: implement a raw.shape() like numpy array. BUt too slow.
# raw.get_channel_types(unique=True) | cubedata=raw.get_data() |
#subraw=raw.copy().pick_types(seeg=True).pick([0,1,2,3,4])  # (110, 648081)
#cropSub=subraw[1,11*1000:13*1000]  # crop 11s to 13s of channel 1, plot: plt.plot(cropSub[1],cropSub[0].T)
#raw.copy().plot(events=events, scalings=dict(seeg=5000))
#raw.copy().pick_types(seeg=True).plot(events=events, scalings=dict(seeg=5000))


## frequency analysis
# PSD: power spectral density
#fig = raw.copy().pick_types(seeg=True).plot_psd(tmin=0,tmax=15, fmin=0.1, fmax=250,average=True)

# Notch filter
# plot some psd before notch. fmax can be ignored
print("Plot psd before and after notch filter.")
raw.plot_psd(tmin=None, tmax=None,picks=['seeg'],average=True,spatial_colors=False,color='black', show=False)
fig=plt.gcf()
ax=plt.gcf().get_axes()
plt.ion()
# notch filter
freqs = (50,150,250,350)
raw.notch_filter(freqs=freqs, picks=['seeg']) # no stim
raw.plot_psd(tmin=None, tmax=None,picks=['seeg'],ax=ax,average=True,spatial_colors=False,color='red')
ax[0].text(0.5,0.9,'Black:before. Red:after.',fontsize=15,transform=fig.transFigure)
figname = plot_dir+'psdWithNotch.png'
fig.savefig(figname,dpi=400)
plt.close(fig)


###  epoching
print("Epoching...")
epochs = mne.Epochs(raw, events, tmin=0, tmax=15,baseline=None)
# extract event according to epochs.event_id
epoch0=epochs['1']
epoch1=epochs['2']
epoch2=epochs['3']
epoch3=epochs['4']
print("Saving epochings...")
if not os.path.exists(data_dir+'PF'+str(sid)+'/data/'):
    os.makedirs(data_dir+'PF'+str(sid)+'/data/')
epoch0.save(data_dir+'PF'+str(sid)+'/data/'+'moveEpoch0.fif', overwrite=True)
epoch1.save(data_dir+'PF'+str(sid)+'/data/'+'moveEpoch1.fif', overwrite=True)
epoch2.save(data_dir+'PF'+str(sid)+'/data/'+'moveEpoch2.fif', overwrite=True)
epoch3.save(data_dir+'PF'+str(sid)+'/data/'+'moveEpoch3.fif', overwrite=True)

print("Plotting force after epoching.")
for i in range(4):
    fig, ax = plt.subplots()
    plt.ion()
    ax.clear()
    epoch=epochs[str(i+1)].get_data()
    targetforce=np.reshape(epoch[:,-2,:],(1,-1))
    realforce=np.reshape(epoch[:,-3,:],(1,-1))
    plt.plot(targetforce[0, :], label='Target')
    plt.plot(realforce[0,:], label='Target')
    plt.title("Force of Epoch "+str(i))
    plt.legend()
    figname = plot_dir + 'forceOfEpoch'+str(i)
    fig.savefig(figname)
    plt.close(fig)
del epochs, epoch0, epoch1, epoch2, epoch3

