import os
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
import numpy as np
import matplotlib.pyplot as plt
from grasp.config import *
import mne

try:
    mne.set_config('MNE_LOGGING_LEVEL', 'ERROR')
except TypeError as err:
    print(err)

test=mne.read_epochs('/Volumes/Samsung_T5/seegData/PF6/data/moveEpoch0.fif')
print(MNE_LOGGING_LEVEL)