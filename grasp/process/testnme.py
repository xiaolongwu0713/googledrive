import os
import numpy as np
import matplotlib.pyplot as plt
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample','sample_audvis_raw.fif')
raw2 = mne.io.read_raw_fif(sample_data_raw_file)

raw2.crop(tmax=60)

eeg_and_eog = raw2.copy().pick_types(eeg=True)
