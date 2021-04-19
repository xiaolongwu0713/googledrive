import numpy as np
MNE_LOGGING_LEVEL='ERROR' # or mne.set_log_level(verbose='ERROR'), then mne.set_log_level(return_old_level=True)

tmp_dir='/tmp/'
root_dir='/Users/long/BCI/python_scripts/googleDrive/' # this is project root
data_raw='/Users/long/BCI/data/grasp_data/PF6_SYF_2018_08_09_Simply/' #raw data and processed data
data_dir='/Users/long/BCI/data/grasp_data/PF6_SYF_2018_08_09_Simply/data/' # preprocessed data
#data_dir='/content/drive/MyDrive/data/' # googleDrive
mode=1
processed_data=data_dir

# Todo: All variable should be indexed begin with 0.
useChannels=np.concatenate((np.arange(0,15),np.arange(16,29),np.arange(37,119)))
activeChannels = [8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 62, 63, 69, 70, 105, 107,108, 109, 110] # 111 is force channel, index start from 1
stim=30 - 1
activeChannels=[chn -1 for chn in activeChannels]
badtrials=[]
badtrials.append([12, 13, 21, 22, 23, 24, 26])
badtrials.append([23, 24, 28])
badtrials.append([3, 4, 8, 11, 16, 21, 22, 23, 24, 25, 29])
badtrials.append([4, 21, 24, 25, 29])
badtrials=[[i-1 for i in sublist] for sublist in badtrials]


fbands=[] #delta, theta, alpha,beta,gamma
fbands.append([0.5, 4])
fbands.append([4, 8])
fbands.append([8, 12])
fbands.append([13, 30])
fbands.append([60, 140])

