import numpy as np


MNE_LOGGING_LEVEL='ERROR' # or mne.set_log_level(verbose='ERROR'), then mne.set_log_level(return_old_level=True)

tmp_dir='/tmp/'
root_dir='/Users/long/BCI/python_scripts/googleDrive/' # this is project root
data_raw='/Volumes/Samsung_T5/seegData/' #raw data and processed data
#data_dir='/Users/long/BCI/data/grasp_data/PF6_SYF_2018_08_09_Simply/data/' # preprocessed data
data_dir='/content/drive/MyDrive/data/' # googleDrive
mode=1
processed_data=data_dir

# Todo: All variable should be indexed begin with 0.
sid=[1,2,6,10,16] #6 is the first subject

# matlab: useChannels=[1:15,17:29,38:119];
# useChannels[sid]=[channel list]
useChannels=[0]*999 # initilize with 999 subject
useChannels[6]=np.concatenate((np.arange(0,15),np.arange(16,29),np.arange(37,119)))

activeChannels=[0]*999
activeChannels[6] = [i-1 for i in [8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 62, 63, 69, 70, 105, 107,108, 109, 110]]# 111 is force channel, index start from 1

stim=30 - 1

# badtrials[sid][trialNum]
badtrials=[[]]*999
badtrials[6].append([i-1 for i in [12, 13, 21, 22, 23, 24, 26]])
badtrials[6].append([i-1 for i in [23, 24, 28]])
badtrials[6].append([i-1 for i in [3, 4, 8, 11, 16, 21, 22, 23, 24, 25, 29]])
badtrials[6].append([i-1 for i in [4, 21, 24, 25, 29]])


fbands=[] #delta, theta, alpha,beta,gamma
fbands.append([0.5, 4])
fbands.append([4, 8])
fbands.append([8, 12])
fbands.append([13, 30])
fbands.append([60, 140])

# some cross module variables, you can import this variable as:
# import grasp.config as myVar, then myVar.preds=...
# OR, just make them global
#preds=[]
#targets=[]

# Lambda is used by skorch get_loss function
Lambda = 1e-6
preds=[]
targets=[]


def printVariables(variable_names):
    for k in variable_names:
        max_name_len = max([len(k) for k in variable_names])
        print(f'  {k:<{max_name_len}}:  {globals()[k]}')

if __name__ == "__main__":
    ks = [k for k in dir() if (k[:2] != "__" and k !='np' and not callable(globals()[k]))]
    #for k in ks:
    #    print(type(k))
    printVariables(ks)

