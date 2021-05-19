'''
question: TF result from below methods are so differeent.
choose a good channel: sid10-channel5
epoch_tfr=tf_morlet(epoch,average=False)
Method 1, avg1=epoch_tfr.average.data --> normalization-->imshow
Method 2, normalization-->average--->imshow
algebra prove that these two methods are different:
method1: mean( 10log(y1/x1), 10log(y2/x2)..., 10log(yN/xN) )
method2: 10log( mean(y1,y2...yN)/mean(x1,x2...xN) )
N is trial number.
when x1=x2=...=xN and y1=y2=...yN, method1=method2.
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


sid=6
# fast testing
#activeChannels[sid]=activeChannels[sid][4]

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
movement=0
singleMovementEpoch=mne.read_epochs(
        data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch'+str(movement)+'.fif').pick(picks=activeChannels[sid])
ch_names=singleMovementEpoch.ch_names
chIndex=5

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

fig,ax=plt.subplots(2)
ax0=ax[0]
ax1=ax[1]
tmp = tfr_morlet(singleMovementEpoch, picks=[chIndex],
                 freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=False,
                                       decim=decim, n_jobs=1)


#Method: average then mornalization
avg1=tmp.average().data.squeeze()
base_avg=np.mean(avg1[:,2500:3250],axis=1)
avg1 = 10*np.log10(avg1 / base_avg[:, None])
im0 = ax0.imshow(avg1, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
ax0.set_aspect('auto')
fig.colorbar(im0, orientation="vertical", fraction=0.046, pad=0.02, ax=ax0)
#Method: normalization then average
avg2_tmp=tmp.data.squeeze()
base = np.mean(avg2_tmp[:,:, 2500:3250],axis=2)
trail_norm = 10*np.log10(avg2_tmp / base[:,:, None])
avg2=np.mean(trail_norm,axis=0)
#Method: vmin and vmax have to shift in order to have the same plot as before.
im1 = ax1.imshow(avg2, origin='lower', cmap='RdBu_r', vmin=vmin-2.5, vmax=vmax-2.5)
ax1.set_aspect('auto')
fig.colorbar(im1, orientation="vertical", fraction=0.046, pad=0.02, ax=ax1)

img=ax1.images
img[0].colorbar.remove()
ax1.clear()

#Question: however this simple simulation seems like both methods should return similar result
x=np.random.randn(1,40)+10
y=np.random.randn(1,40)+5
m1=1/40*10*np.log10((np.prod(x))/(np.prod(y)))
m2=10*np.log10((np.mean(x))/(np.mean(y)))








