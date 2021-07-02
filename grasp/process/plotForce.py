import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

import hdf5storage
import mne
import numpy as np
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell
from grasp.config import *
from grasp.process.utils import get_trigger, getMovement, get_trigger_normal, getForceData, \
    genSubTargetForce, getRawData
import matplotlib.pyplot as plt
from grasp.process.channel_settings import *

sid=6
forceChannel=-3
stimChannel=-1
plot_dir=data_dir + 'PF' + str(sid) +'/process/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Note: input: session in [0,1,2,3]; channels=activeChannels or useChannels
channels=[forceChannel,stimChannel] # real force and stim


movements=4
forces=[]
stims=[]
for movement in range(movements):
    tmp=mne.read_epochs(data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch'+str(movement)+'.fif')
    tmp.pick(picks=channels)
    force_tmp = tmp.copy().pick(picks=[0])
    stim_tmp = tmp.copy().pick(picks=[1])
    forces.append(force_tmp)
    stims.append(stim_tmp)


force_stat=[] # force_stat=[oneMovement][0] is mean force value. force_stat=[oneMovement][1] is std.
fig,ax=plt.subplots(2,2)
for i in range(movements):
    force_stat.append([])
    tmp=np.squeeze(forces[i].get_data()) #(40, 15001)
    tmp_mean=np.mean(tmp,axis=0)
    mean_length=tmp_mean.shape[0]
    tmp_std=np.std(tmp,axis=0)
    force_stat[i].append(tmp_mean)
    force_stat[i].append(tmp_std)
    ax[i // 2][i - (i // 2) * 2].plot(force_stat[i][0])  # plot mean
    ax[i // 2][i - (i // 2) * 2].fill_between(range(mean_length), force_stat[i][0] - force_stat[i][1],
                                              force_stat[i][0] + force_stat[i][1], alpha=0.2)
    ax[i // 2][i - (i // 2) * 2].text(0.8, 0.8, 'M'+ str(i+1), fontsize=10,transform=ax[i // 2][i - (i // 2) * 2].transAxes)

figname = plot_dir + 'force_stats.pdf'
fig.savefig(figname, dpi=400)
#plt.savefig(figname, format='eps')
#fig.savefig(figname, format='eps')
plt.close(fig)
