import numpy as np
import torch
import mne
import socket
import os, re,sys

try:
    mne.set_config('MNE_LOGGING_LEVEL', 'ERROR')
except TypeError as err:
    print('error happens.')
    print(err)

tmp_dir='/tmp/'

if socket.gethostname() == 'longsMac':
    #sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
    if os.path.exists('/Volumes/Samsung_T5/data/gesture/'):
        data_dir='/Volumes/Samsung_T5/data/gesture/'

    else:
        data_dir = '/Users/long/Documents/data/gesture/'# temp data dir
    info_dir = '/Users/long/Documents/data/gesture/info/'  # sub info
    #tmp_data_dir='/Users/long/Documents/data/gesture/'
    root_dir = '/Users/long/BCI/python_scripts/googleDrive/'  # this is project root

elif socket.gethostname() == 'workstation':
    #sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
    #data_dir = 'C:/Users/wuxiaolong/Desktop/BCI/data/gesture/'  # temp data dir
    data_dir = 'H:/Long/data/gesture/'  # temp data dir
    info_dir = 'H:/Long/data/gesture/preprocessing/'  # temp data dir

location=os.getcwd()
if re.compile('/content/drive').match(location):
    data_dir='/content/drive/MyDrive/data/gesture/' # googleDrive
    root_dir='/content/drive/MyDrive/' # googleDrive
    info_dir = '/content/drive/MyDrive/data/gesture/preprocessing/'

# paradigm definition
# channel:  SEEG+EMG+class

default_frequency=1000

# participants details
classNum = 5
# Frequencies = np.array([[2, 1000], [3, 1000], [ 4, 1000], [5, 1000], [ 7, 1000], [ 8, 1000], [ 9, 1000], [ 10, 2000],
#     [13, 2000], [ 16, 2000], [ 17, 2000], [ 18, 2000], [ 19, 2000], [ 20, 1000], [ 21, 1000], [ 22, 2000], [ 23, 2000],  #[24, 2000], [ 25, 2000], [ 26, 2000],
#      [  29, 2000], [ 30, 2000], [ 31, 2000], [ 32, 2000], [ 34, 2000], [ 35, 1000],
#     [36, 2000], [ 37, 2000], [41, 2000]])
Frequencies = np.load(info_dir+'Info.npy', allow_pickle=True)

# [sid, fs, number of electrodes]
Electrodes=np.array([[2, 1000,121], [3, 1000,180], [ 4, 1000,60], [5, 1000,178], [ 7, 1000,143],[ 8, 1000,169],
    [ 9, 1000,114], [ 10, 2000,208],[11,500,194],[12,500,94],[13, 2000,102], [14, 2000,130],[ 16, 2000,170],
    [ 17, 2000,144], [ 18, 2000,144], [ 19, 2000,137], [ 20, 1000,108], [ 21, 1000,118], [ 22, 2000,150],
    [ 23, 2000,198],[24, 2000,130],[ 25, 2000,137], [ 26, 2000,154],[  29, 2000,110], [ 30, 2000,108], [ 31, 2000,72],
    [ 32, 2000,56], [ 34, 2000,102], [ 35, 1000,136],[36, 2000,117],[ 37, 2000,64],[38,2000,242],[39, 2000,126], [41, 2000,190]])

active_channels={}
active_channels['10']=range(162,168,1)

fbands=[] #delta, theta, alpha,beta,gamma
fbands.append([0.5, 4])
fbands.append([4, 8])
fbands.append([8, 12]) # mu(motor cortex)/alpha(visual cortex)
fbands.append([13, 30]) # beta
fbands.append([60, 125]) # genBandPower_znormalied.py
#fbands.append([60, 140]) # gamma: 55-85

ERD=[13,30]
ERS=[55,100]

def printVariables(variable_names):
    for k in variable_names:
        max_name_len = max([len(k) for k in variable_names])
        print(f'  {k:<{max_name_len}}:  {globals()[k]}')

if __name__ == "__main__":
    ks = [k for k in dir() if (k[:2] != "__" and k !='np' and not callable(globals()[k]))]
    #for k in ks:
    #    print(type(k))
    printVariables(ks)

