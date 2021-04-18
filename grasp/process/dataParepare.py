import mne
from grasp.config import activeChannels
import numpy as np

basedir='/Users/long/Documents/BCI/python_scripts/grasp/process/'
move1=mne.read_epochs('/Users/long/BCI/python_scripts/grasp/process/move1epoch.fif')
move2=mne.read_epochs('/Users/long/BCI/python_scripts/grasp/process/move2epoch.fif')
move3=mne.read_epochs('/Users/long/BCI/python_scripts/grasp/process/move3epoch.fif')
move4=mne.read_epochs('/Users/long/BCI/python_scripts/grasp/process/move4epoch.fif')

activeChannels=[item -1 for item in activeChannels] + [-2,-1]
move1=move1.get_data(picks=['seeg','emg'])[:,activeChannels,:] # (40, 21, 15001)
move2=move2.get_data(picks=['seeg','emg'])[:,activeChannels,:] # (40, 21, 15001)
move3=move3.get_data(picks=['seeg','emg'])[:,activeChannels,:] # (40, 21, 15001)
move4=move4.get_data(picks=['seeg','emg'])[:,activeChannels,:] # (40, 21, 15001)

np.save('/Users/long/Documents/BCI/python_scripts/grasp/move1epoch.npy',move1)
np.save('/Users/long/Documents/BCI/python_scripts/grasp/move2epoch.npy',move2)
np.save('/Users/long/Documents/BCI/python_scripts/grasp/move3epoch.npy',move3)
np.save('/Users/long/Documents/BCI/python_scripts/grasp/move4epoch.npy',move4)


