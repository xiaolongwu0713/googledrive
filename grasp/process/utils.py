import hdf5storage
import numpy as np
import os
import scipy.io
from scipy import signal

from grasp.config import raw_data, processed_data, mode
from grasp.process.config import result_dir

## usage: aa=loadData(31,1,'good'/'bad'/'power'/'epoch61'), then load the data part by checking aa.keys()
def loadData(pn, session, *args):
    arg=args[0]
    if len(args)==0:
        filename = os.path.join(raw_data,'P'+str(pn), '1_Raw_Data_Transfer',
                                'P'+str(pn)+'_H'+str(mode)+'_'+str(session)+'_Raw.mat')
    else:
        if arg=='good':
            filename=os.path.join(processed_data,'P'+str(pn),
                                  'P'+str(pn)+'_H'+str(mode)+'_'+str(session)+'_goodLineChn.mat')
        elif arg=='bad':
            filename = os.path.join(processed_data, 'P' + str(pn),
                                    'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_goodLineChn.mat')
        elif arg == 'discard':
            filename = os.path.join(processed_data, 'P' + str(pn),
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_discardChn.mat')
        elif arg == 'power':
            filename = os.path.join(processed_data, 'P' + str(pn),
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_power.mat')
        elif arg == 'events':
            filename = os.path.join(processed_data, 'P' + str(pn),'eeglabData',
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_eventtable.txt')
            tmp=a=np.loadtxt(filename)
            return tmp
        elif arg == 'eventave':
            filename = os.path.join(processed_data, 'P' + str(pn),'eeglabData',
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_eventtable_ave.mat')
        elif arg == 'eventave':
            filename = os.path.join(processed_data, 'P' + str(pn),'eeglabData',
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_eventtable_ave.mat')
        elif re.compile('epoch').match(arg):
            filename=os.path.join(processed_data, 'P' + str(pn),'eeglabData',
                                'P' + str(pn) + '_H' + str(mode) + '_' + str(session) + '_' + arg+'.mat')
    mat = scipy.io.loadmat(filename)
    return mat  # return np arrary. avedata is the key of this dict, data dim: eles,time,trials



def get_trigger(triggerChannel):
    from sklearn.preprocessing import MinMaxScaler
    triggerTmp = np.zeros((triggerChannel.shape[0]))
    triggerChannel[abs(triggerChannel) > 10000] = 0
    triggerChannel[abs(triggerChannel) < 2000] = 0
    triggerChannel[triggerChannel < 0] = 0
    triggerTmp[triggerChannel > 2000] = 100
    prev_trigger = 254765
    next_trigger = 284887
    estimate = int((prev_trigger + next_trigger) / 2)
    triggerTmp[prev_trigger + 1:next_trigger - 1] = 0
    triggerTmp[estimate] = 100
    tindex = np.nonzero(triggerTmp)[0]  # nonzero returns tuple

    trigger=np.zeros((triggerChannel.shape[0]))
    index = []
    for i in range(tindex.shape[0]):
        if i == 0:
            index.append(tindex[0])
        if (tindex[i] - tindex[i - 1]) > 2000:
            index.append(tindex[i])
    trigger[index]=100
    lastone=np.nonzero(trigger)[0][-1]
    trigger[lastone]=0
    return trigger

def get_trigger_normal(triggerChannel):
    triggerTmp = np.zeros((triggerChannel.shape[0]))
    triggerChannel[abs(triggerChannel) > 10000] = 0
    triggerChannel[abs(triggerChannel) < 2000] = 0
    triggerChannel[triggerChannel < 0] = 0
    triggerTmp[triggerChannel > 2000] = 100

    tindex = np.nonzero(triggerTmp)[0]  # nonzero returns tuple
    trigger=np.zeros((triggerChannel.shape[0]))
    index = []
    for i in range(tindex.shape[0]):
        if i == 0:
            index.append(tindex[0])
        if (tindex[i] - tindex[i - 1]) > 2000:
            index.append(tindex[i])
    trigger[index]=100
    lastone = np.nonzero(trigger)[0][-1]
    trigger[lastone] = 0
    return trigger

def genSubTargetForce(type):
    aNum = 0.05
    if type==1:
        flevel = 0.4
        prep = 2000
        ascendDuration = 3000
        holding = 2500
        y1 = np.ones((1, prep)) * aNum
        y2 = (flevel - aNum) / ascendDuration * np.arange(1,ascendDuration+1) + aNum
        y3 = flevel * np.ones((1, holding))
        y4 = np.ones((1, (15000 - prep - ascendDuration - holding))) * aNum
        subtarget = np.concatenate((y1, y2[np.newaxis,:], y3, y4), axis=1) # (1, 15000)
    elif type==2:
        flevel = 1.0
        prep = 2000
        ascendDuration = 9000
        holding = 2500
        y1 = np.ones((1, prep)) * aNum
        y2 = (flevel - aNum) / ascendDuration * np.arange(1, ascendDuration + 1) + aNum
        y3 = flevel * np.ones((1, holding))
        y4 = np.ones((1, (15000 - prep - ascendDuration - holding))) * aNum
        subtarget = np.concatenate((y1, y2[np.newaxis, :], y3, y4), axis=1)  # (1, 15000)
    elif type==3:
        flevel = 0.4
        prep = 2000
        ascendDuration = 1000
        holding = 2500
        y1 = np.ones((1, prep)) * aNum
        y2 = (flevel - aNum) / ascendDuration * np.arange(1, ascendDuration + 1) + aNum
        y3 = flevel * np.ones((1, holding))
        y4 = np.ones((1, (15000 - prep - ascendDuration - holding))) * aNum
        subtarget = np.concatenate((y1, y2[np.newaxis, :], y3, y4), axis=1)  # (1, 15000)
    elif type==4:
        flevel = 1.0
        prep = 2000
        ascendDuration = 3000
        holding = 2500
        y1 = np.ones((1, prep)) * aNum
        y2 = (flevel - aNum) / ascendDuration * np.arange(1, ascendDuration + 1) + aNum
        y3 = flevel * np.ones((1, holding))
        y4 = np.ones((1, (15000 - prep - ascendDuration - holding))) * aNum
        subtarget = np.concatenate((y1, y2[np.newaxis, :], y3, y4), axis=1)  # (1, 15000)
    return  subtarget

def readMat(filename):
    import hdf5storage
    return hdf5storage.loadmat(filename)
def getRawData(seegfile,useChannels):
    mat = hdf5storage.loadmat(seegfile)
    myraw = mat['Data'][useChannels, :]  # (110, 1296162)
    myraw = signal.decimate(myraw, 2, axis=1,ftype='iir',zero_phase=True)  # (110, 648081)
    triggerRaw = signal.decimate(mat['Data'][29, :], 2)
    fs = int(mat['Fs'][0][0] / 2)  # 1000
    chnRaw = mat['ChnName']
    channels = np.asarray([chnRaw[i][0][0][0] for i in range(len(chnRaw))])  # list with len=126
    channels = channels[useChannels]  # (110,)
    ch_names = [channelsName.strip() for channelsName in channels]
    return myraw, triggerRaw, ch_names

def getMovement(triggerfile):
    mat = hdf5storage.loadmat(triggerfile)
    # trigger_time=mat['Info']['task_time'][0][0] #task duration, e.g 5.5,7.5, 13.5 s. np array, shape: (1,40)
    task_len = mat['Info']['task_time'][0][0]
    trial_len = mat['Info']['trial_length'][0][0]  # trial duration: all 15 s. np array, shape: (1,40)
    xaxis = mat['Info']['Xaxis'][0][0]  # (1, 40) each element has 4 values
    yaxis = mat['Info']['Yaxis'][0][0]  # (1, 40) each element has 4 values
    movement = mat['Info']['Exp_Seq'][0][0]  # trial type, e.g 1,2,3,4. (1, 40) movement type
    return movement

def getForceData(forcefile,trigger):
    import matplotlib.pyplot as plt
    mat = hdf5storage.loadmat(forcefile)
    key=list(mat.keys())[-1] # keys are different for different file
    forceTmp = mat[key]  # (6039000, 1)
    force = forceTmp[1::2]
    timepoints = forceTmp[0::2]
    ffs = int(timepoints.shape[0] / timepoints[-1])  # 5000
    resample=int(ffs/1000)
    force = signal.decimate(force, resample, axis=0)  # (603900, 1), padding force to aligh with trigger data.
    force = force - min(force)
    paddingBefore = np.nonzero(trigger)[0][0]  # % 28206
    paddingEnd = trigger.shape[0] - force.shape[0] - paddingBefore
    firstpoint=force[0,0]
    lastpoint=force[-1,0]
    force = np.concatenate( (np.ones((1, paddingBefore))*firstpoint, force.T, np.ones((1, paddingEnd))*lastpoint), axis=1).T  # (648081, 1)

    return force.T


def add_arrows(axes):
    # add some arrows at 60 Hz and its harmonics
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (60, 120, 180, 240):
            idx = np.searchsorted(freqs, freq)
            # get ymax of a small region around the freq. of interest
            y = psds[(idx - 4):(idx + 5)].max()
            ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                     width=0.1, head_width=3, length_includes_head=True)
