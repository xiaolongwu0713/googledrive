import hdf5storage
import mne
import numpy as np
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell
from grasp.config import activeChannels, stim,badtrials,data_raw,data_dir,fbands
from grasp.process.utils import get_trigger, getMovement, get_trigger_normal, getForceData, \
    genSubTargetForce, getRawData
import matplotlib.pyplot as plt

plot_dir = 'grasp/process/genBandPower_result/'

# Note: input: session in [0,1,2,3]; channels=activeChannels or useChannels
session = 0
channels=activeChannels
#activeChannels.append(stim) # 29 is stim(trigger) channel
#activeChannels.sort()

print("Read raw data from disk.")
seegfile=data_raw+'SEEG_Data/PF6_F_'+str(session+1)+'.mat'
raw, triggerRaw,ch_names=getRawData(seegfile,channels)
fs=1000

# format trigger
print("Format trigger channel.")
movementfile=data_raw+'Trigger_Data/Trigger_Information_'+str(session+1)+'.mat' # movement code series in [1,2,3,4]
movement=getMovement(movementfile)
if session == 0:
    trigger = get_trigger(triggerRaw)
    tindex = np.nonzero(trigger)[0]
    trigger[tindex] = movement
else:
    trigger = get_trigger_normal(triggerRaw)
    tindex = np.nonzero(trigger)[0]
    trigger[tindex] = movement

### 5, real force data
print("Read real force data.")
forcefile=data_raw + 'Force_Data/SYF-' + str(session+1) + '-' + str(session+2) +'.mat'
force = getForceData(forcefile, trigger)

### 6 generate the target force
print("Generate target force data.")
targetForce=np.ones((trigger.shape[0]))*0.05 # (648081,)
indexAll=np.nonzero(trigger)[0] # 40
for (index,j) in zip(indexAll,np.arange(indexAll.shape[0])):
    targetForce[index:index+15000]=genSubTargetForce(movement[0][j])

# channels info
ch_names=np.append(ch_names,['force','target','stimulation']) # channel name can't be equal to type
ch_types=np.concatenate((np.repeat(np.array('seeg'),len(activeChannels)),['emg','emg','stim']),axis=0)

# stack raw with real force, target force and trigger
print("Stack raw, real froce, target force and trigger data together.")
raw=np.concatenate((raw,force,targetForce[np.newaxis,:],trigger[np.newaxis,:]),axis=0)
info = mne.create_info(ch_names=list(ch_names), ch_types=list(ch_types), sfreq=fs)
raw = mne.io.RawArray(raw, info)
events = mne.find_events(raw, stim_channel='stimulation',consecutive=False)

# plot some psd before notch. fmax can be ignored
print("Plot psd before and after notch filter.")
raw.plot_psd(tmin=None, tmax=None,picks=[0],average=False,spatial_colors=False,color='black', show=False)
fig=plt.gcf()
ax=plt.gcf().get_axes()
plt.ion()
# notch filter
freqs = (150,250,350,450)
raw.notch_filter(freqs=freqs, picks=['seeg']) # no stim
raw.plot_psd(tmin=None, tmax=None,picks=[0],ax=ax,average=False,spatial_colors=False,color='red')
ax[0].text(0.5,0.9,'Black:before. Red:after.',fontsize=15,transform=fig.transFigure)
figname = root+plot_dir + str(session)+'psdOfchannel0WithNotch.png'
fig.savefig(figname,dpi=400)
plt.close(fig)

# TODO : try other filter method, because the psd show very large transition bandwidth
# band pass data: fbands includes: delta, theta, alpha,beta,gamma
print("5 bandpass filter....")
bandsName=['delta','theta','alpha','beta','gamma']
delta=raw.copy().pick(picks=['seeg']).filter(l_freq=fbands[0][0], h_freq=fbands[0][1]) # (19, 648081)
theta=raw.copy().pick(['seeg']).filter(l_freq=fbands[1][0], h_freq=fbands[1][1]) # ..
alpha=raw.copy().pick(picks=['seeg']).filter(l_freq=fbands[2][0], h_freq=fbands[2][1]) # ..
beta=raw.copy().pick(picks=['seeg']).filter(l_freq=fbands[3][0], h_freq=fbands[3][1]) # ..
gamma=raw.copy().pick(picks=['seeg']).filter(l_freq=fbands[4][0], h_freq=fbands[4][1]) # ..
bands=[delta,theta,alpha,beta,gamma]
# some plots
print("Plot psd after bandpass.")
fig,ax=plt.subplots()
delta.plot_psd(picks=[0,],ax=ax,xscale='log',average=False,spatial_colors=False,color='yellow')
fig=plt.gcf()
ax=plt.gcf().get_axes()
theta.plot_psd(tmin=None, tmax=None,picks=[0],ax=ax,xscale='log',average=False,spatial_colors=False,color='black')
alpha.plot_psd(tmin=None, tmax=None,picks=[0],ax=ax,xscale='log',average=False,spatial_colors=False,color='green')
beta.plot_psd(tmin=None, tmax=None,picks=[0],ax=ax,xscale='log',average=False,spatial_colors=False,color='red')
gamma.plot_psd(tmin=None, tmax=None,picks=[0],ax=ax,xscale='log',average=False,spatial_colors=False,color='blue')
ax[0].text(0.5,0.9,'psd after 5 passbands on one random channel',fontsize=15,transform=fig.transFigure)
figname = root+plot_dir + str(session)+'psdOfchannel0With5Passbands.png'
fig.savefig(figname,dpi=400)
plt.close(fig)

# Apply hilbert
fig,ax=plt.subplots()
ax.plot(alpha.copy().pick(picks=[0,])[0][0][0,:5000]) #.plot(events=events,scalings=5e+3)
ax.plot(alpha.copy().pick(picks=[0,]).apply_hilbert(envelope=True)[0][0][0,:5000]) #.plot(events=events,scalings=5e+3,ax=ax)
plt.title('Hilbert of alpha segment')
figname= root+plot_dir + str(session)+'hilbertOfAlphaSegment.png'
fig.savefig(figname,dpi=400)
plt.close(fig)
# Question: how to use apply_function??
# Todo: calcualte power
#a=alpha.copy().pick(picks=[0,]).apply_hilbert(envelope=True).crop(tmin=0,tmax=0.1)
#np_power_arg=[2]
#b=a.copy().apply_function(np.power,args=np_power_arg)
print("Applying hilbert.")
for i in range(len(bands)):
    bands[i].apply_hilbert(envelope=True)


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

# stack all bands into one raw
print("Stacking all 5 bands together.")
for i in range(len(bands)):
    for c in bands[i].ch_names:
        bands[i].rename_channels({c: bandsName[i] + '_' + c}) # change the ch_name, otherwise no way to concatenate
raw_bak=raw.copy()
raw.add_channels([band.load_data() for band in bands],force_update_info=True) # original 22 chs + 5*19 subband chns

# sanity check
print("Plot force, target force and trigger.")
forces=raw.copy().pick(picks=['emg']).get_data() #(2, 648081)
trigger=raw.copy().pick(picks=['stim']).get_data() # (1, 648081)
fig,ax=plt.subplots()
plt.plot(forces[0,:],label='real force',linewidth=0.2)
plt.plot(forces[1,:],label='target force',linewidth=0.2)
plt.plot(trigger[0,:],label='trigger',linewidth=0.2)
figname = root+plot_dir + str(session)+'forceAndTriggerInfo.png'
fig.savefig(figname,dpi=400)
plt.close(fig)

###  epoching
print("Epoching...")
epochs = mne.Epochs(raw, events, tmin=0, tmax=15,baseline=None)
# extract event according to epochs.event_id
epoch1=epochs['1']
epoch2=epochs['2']
epoch3=epochs['3']
epoch4=epochs['4']
print("Saving epochings...")
epoch1.save(data_dir+'s'+str(session)+'move0BandsEpoch.fif', overwrite=True)
epoch2.save(data_dir+'s'+str(session)+'move1BandsEpoch.fif', overwrite=True)
epoch3.save(data_dir+'s'+str(session)+'move2BandsEpoch.fif', overwrite=True)
epoch4.save(data_dir+'s'+str(session)+'move3BandsEpoch.fif', overwrite=True)
print("Finish.")