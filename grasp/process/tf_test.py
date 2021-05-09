import mne
import numpy as np
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell

from grasp.config import data_dir
from grasp.utils import activeChannels, badtrials
import matplotlib.pyplot as plt

sid=6
chooseOneMovement=0
#epochs = mne.read_epochs('/Users/long/BCI/python_scripts/grasp/process/move1epoch.fif', preload=True)
epochs = mne.read_epochs(data_dir + 'PF' + str(sid) + '/data/' + 'moveEpoch'+str(chooseOneMovement)+'.fif', preload=True)


# NOTE: common operation: oneepoch=epochs[0]
#cube=epochs.get_data() # (40, 113, 15001)
#epochs.plot(n_epochs=10,scalings=dict(seeg=5000))
#shorter_epochs = epochs.copy().crop(tmin=-1, tmax=4, include_tmax=True)

## visuallization
#epochs['move1'].plot_image(picks='seeg', combine='mean')

## frequency analysis
# define frequencies of interest (log-spaced)
freqs = np.logspace(*np.log10([55, 150]), num=80)
n_cycles = freqs / 2.  # different number of cycle per frequency
power= tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,return_itc=False, decim=4, n_jobs=1)
chn=15
power.plot([chn], baseline=(-0.5, 0), mode='logratio', title=power.ch_names[chn])

# plot all power on one figure
fig,axes=plt.subplots(ncols=5,nrows=5,figsize=(10,10),)
for chn in np.arange(0,24):
    row=chn//5
    power.plot([chn], baseline=(-0.5, 0), mode='logratio', title=power.ch_names[chn], axes=axes[row][chn-(row)*5])
#### plot power of certain frequency range on channel-vs-time plot


# method: morlet
#freqs = np.logspace(*np.log10([2, 400]), num=200)
freqs = np.linspace(2,300, num=150)
n_cycles = np.linspace(5,160, num=150)  # different number of cycle per frequency
power=mne.time_frequency.read_tfrs('/Users/long/BCI/python_scripts/grasp/process/tfr1')[0]
power = tfr_morlet(epochs, picks=[7], freqs=freqs, n_cycles=n_cycles, use_fft=True,return_itc=False, decim=3, n_jobs=1)
power.save('/Users/long/BCI/python_scripts/grasp/process/tfr_morlet1', overwrite=True)
# mode: ‘mean’ | ‘ratio’ | ‘logratio’ | ‘percent’ | ‘zscore’ | ‘zlogratio’
# add cmap=('summer',True) parameter and up/down arrow to pick your color map
power.plot([0], baseline=(13,14.5), mode='zscore', vmin=-4,vmax=4)

# method: multitaper method.
# More freq smothing with larger time_bandwidth
time_bandwidth=4.0
freqs = np.linspace(2,300, num=150)
# different number of cycle for each frequency.
# More time smothing whei larger cycles.
n_cycles = np.linspace(10,160, num=150)
# time smothing: cycle/frequency = timeWidow
# freq smothing: time_bandwidth=2.0= timeWindow * bandwidth, so frequency smothing(bandwidth)=time_bandwidth/(timeWindow)
power = tfr_multitaper(epochs, picks=[7],freqs=freqs, n_cycles=n_cycles,time_bandwidth=time_bandwidth, return_itc=False)
power.plot([0], baseline=(13,14.5), mode='zscore', vmin=-4,vmax=4)
fig,axs=plt.subplots(2,3,figsize=(15,5),sharey=True)
ass=np.asarray(axs).reshape(1,-1)
for type, ax in zip(['mean','ratio','logratio','percent','zscore','zlogratio'],ass.T):
    power.plot([0], baseline=(13, 14), mode=type, axes=ax, vmin=-4, vmax=4)

# method: stockwell method
epochs.load_data()
epoch7=epochs.pick([7])
power = tfr_stockwell(epoch7, fmin=2, fmax=300, width=0.3) # no picks argument in stockwell
power.plot([0], baseline=(13,14), mode='zscore',vmin=-4,vmax=4, show=False)
