from mne.time_frequency import tfr_morlet
from grasp.process.utils import get_trigger, genSubTargetForce, getRawData, getMovement, getForceData, \
    get_trigger_normal
from grasp.config import sid
import numpy as np
import mne
import matplotlib.pyplot as plt
from grasp.process.config import result_dir

# first subject: sid=6
sid=6

# info
# useChannels=[1:15,17:29,38:119]
useChannels=np.concatenate((np.arange(0,15),np.arange(16,29),np.arange(37,119)))
sampling_rate = 1000

## load channel
seegfile={}
triggerfile={}
forcefile={}
seegfile[0]='/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/PF6_F_1.mat'
seegfile[1]='/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/PF6_F_2.mat'
seegfile[2]='/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/PF6_F_3.mat'
seegfile[3]='/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/PF6_F_4.mat'
triggerfile[0]='/Users/long/Documents/BCI/matlab_scripts/force/data/Trigger_Data/Trigger_Information_1.mat'
triggerfile[1]='/Users/long/Documents/BCI/matlab_scripts/force/data/Trigger_Data/Trigger_Information_2.mat'
triggerfile[2]='/Users/long/Documents/BCI/matlab_scripts/force/data/Trigger_Data/Trigger_Information_3.mat'
triggerfile[3]='/Users/long/Documents/BCI/matlab_scripts/force/data/Trigger_Data/Trigger_Information_4.mat'
#emgS1file='/Users/long/Documents/BCI/matlab_scripts/force/data/HDEMG_Data/SYF_session1.mat'
forcefile[0]='/Users/long/Documents/BCI/matlab_scripts/force/data/Force_Data/SYF-1-2.mat'
forcefile[1]='/Users/long/Documents/BCI/matlab_scripts/force/data/Force_Data/SYF-2-3.mat'
forcefile[2]='/Users/long/Documents/BCI/matlab_scripts/force/data/Force_Data/SYF-3-4.mat'
forcefile[3]='/Users/long/Documents/BCI/matlab_scripts/force/data/Force_Data/SYF-4-5.mat'

### 1, raw data
triggerRaw={}
myraw={}
print("Reading raw data.")
for i in range(4):
    myraw[i], triggerRaw[i],chn_names=getRawData(seegfile[i],useChannels)
### 2, experiment setting
movement={}
for i in range(4):
    movement[i]=getMovement(triggerfile[i])
allMovement=np.concatenate((movement[0],movement[1],movement[2],movement[3]),axis=1)
### 4 format the trigger channel
trigger={}
for i in range(4):
    if i==0:
        trigger[i]=get_trigger(triggerRaw[i])
        tindex = np.nonzero(trigger[i])[0]
        trigger[i][tindex] = movement[i]
    else:
        trigger[i] = get_trigger_normal(triggerRaw[i])
        tindex = np.nonzero(trigger[i])[0]
        trigger[i][tindex] = movement[i]

### 5, force data
force={}
for i in range(4):
    print (i)
    force[i] = getForceData(forcefile[i], trigger[i])

### 6 generate the target force
targetForce={}
for i in range(4):
    targetForce[i]=np.ones((trigger[i].shape[0]))*0.05 # (648081,)
    indexAll=np.nonzero(trigger[i])[0] # 40
    for (index,j) in zip(indexAll,np.arange(indexAll.shape[0])):
        targetForce[i][index:index+15000]=genSubTargetForce(movement[i][0][j])

print("Plotting of force, target and trigger.")
for i in range(4):
    fig, ax = plt.subplots()
    plt.ion()
    ax.clear()
    ax.plot(trigger[i][:,], label='Trigger', linewidth=0.1) # (648081,)
    ax.plot(force[i][0,:], label='Force',linewidth=0.1) # (1, 648081)
    ax.plot(targetForce[i][:,], label='TargetForce',linewidth=0.1) #(648081,)
    plt.legend()
    figname = result_dir + 'triggerAndForceAndTarget' + str(i) + '.png'
    fig.savefig(figname,dpi=400)
    plt.close(fig)


## 7 concatenate all 4 session
print("Concatenate data from all 4 sessions.")
rawOf4=np.concatenate((myraw[0],myraw[1],myraw[2],myraw[3]),axis=1) #(110, 2742378)
del myraw
forceOf4=np.concatenate((force[0],force[1],force[2],force[3]),axis=1) #(1, 2742378)
targetForceOf4=np.concatenate((targetForce[0],targetForce[1],targetForce[2],targetForce[3])) #(2742378,)
triggerOf4=np.concatenate((trigger[0],trigger[1],trigger[2],trigger[3])) #(2742378,)

print("Plotting of force for all 4 session.")
fig, ax = plt.subplots()
plt.ion()
ax.clear()
ax.plot(triggerOf4[:,], label='Trigger', linewidth=0.01) # (648081,)
ax.plot(forceOf4[0,:], label='Force',linewidth=0.01) # (1, 648081)
ax.plot(targetForceOf4[:,], label='TargetForce',linewidth=0.01) #(648081,)
plt.legend()
figname = result_dir + 'ALLtriggerAndForceAndTarget.pdf'
fig.savefig(figname)
plt.close(fig)


### 7 concatenat: seeg data + real force + target force + trigger
myraw=np.concatenate((rawOf4,forceOf4, targetForceOf4[np.newaxis,:],triggerOf4[np.newaxis,:]),axis=0) #(113, 648081)
del rawOf4


### create info and raw data
ch_names=np.append(chn_names,['force','target','stim']) # events, emg. total 112 channels = 110+2
#ch_types=np.repeat(np.array('seeg'),126)
ch_types=np.concatenate((np.repeat(np.array('seeg'),110),np.repeat(np.array('emg'),2),np.repeat(np.array('stim'),1)))
info = mne.create_info(ch_names=list(ch_names), ch_types=list(ch_types), sfreq=sampling_rate)
raw = mne.io.RawArray(myraw, info)

### events
# TODO: how to load events from file
#myevents=loadData(31,1,'events')
#event_dict = {'move1': 1, 'move2': 2, 'move3': 3,'move4': 4, 'move5': 5}
# nme can find event from channels of raw data
events = mne.find_events(raw, stim_channel='stim',consecutive=False)
allMovement2=events[:,2]
if (allMovement2==allMovement).all():
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

## notch filter
chn = mne.pick_types(raw.info, seeg=True)
freqs = (50, 150, 250)
raw.notch_filter(freqs=freqs, picks=chn)
#raw_notch = raw.copy().notch_filter(freqs=freqs, picks=chn)
# compare psd before and after notch filter
#for title, data in zip(['Un', 'Notch '], [raw, raw_notch]):
#    fig = data.plot_psd(fmax=250, average=True)
#    fig.subplots_adjust(top=0.85)
#    fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
#     add_arrows(fig.axes[:2])

###  epoching
print("Epoching...")
epochs = mne.Epochs(raw, events, tmin=0, tmax=15,baseline=None)
# extract event according to epochs.event_id
epoch1=epochs['1']
epoch2=epochs['2']
epoch3=epochs['3']
epoch4=epochs['4']
print("Saving epochings...")
epoch1.save('/Users/long/BCI/python_scripts/grasp/process/move1epoch.fif', overwrite=True)
epoch2.save('/Users/long/BCI/python_scripts/grasp/process/move2epoch.fif', overwrite=True)
epoch3.save('/Users/long/BCI/python_scripts/grasp/process/move3epoch.fif', overwrite=True)
epoch4.save('/Users/long/BCI/python_scripts/grasp/process/move4epoch.fif', overwrite=True)

print("Plotting separate Epoch of 4 movements.")
for i in range(4):
    fig, ax = plt.subplots()
    plt.ion()
    ax.clear()
    epoch=epochs[i].get_data()
    targetforce=np.reshape(epoch[:,111,:],(1,-1))
    realforce=np.reshape(epoch[:,110,:],(1,-1))
    plt.plot(targetforce[0, :], label='Target')
    plt.plot(realforce[0,:], label='Target')
    plt.title("Force of Epoch "+str(i))
    plt.legend()
    figname = result_dir + 'forceOfEpoch'+str(i)
    fig.savefig(figname)
    plt.close(fig)
del epochs, epoch1, epoch2, epoch3, epoch4

