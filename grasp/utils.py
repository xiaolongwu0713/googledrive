import random

import mne
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from grasp.config import *
import os
import numpy as np
import scipy.io
import re
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

import sys, importlib
importlib.reload(sys.modules['grasp.config'])
from grasp.config import activeChannels,badtrials,data_dir
from grasp.config import data_raw

def parameterNum(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SEEGDataset1(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor
    def __init__(self, x_tensor, y_tensor, T, step):
        self.x = x_tensor
        self.y = y_tensor
        self.chnNum=x_tensor.shape[1]
        self.T=T
        self.step=step
        self.totalLen = x_tensor.shape[2]  # ms
        self.batch_size = int((self.totalLen - T) / step)  # 280
        assert self.x.shape[0] == self.y.shape[0]
    def __getitem__(self, index):
        trainx=self.x[index]
        trainy=self.y[index]
        x = np.zeros((self.batch_size, 1, self.chnNum, self.T))  # 4D:(280,1,19,1000ms):(batch_size, planes, height, weight)
        y = np.zeros((self.batch_size, 1))  # (280, 1)
        yd = np.zeros((self.batch_size, 1))  # (280, 1)
        for bs in range(self.batch_size):
            x[bs, 0, :, :] = trainx[:, bs*self.step:(bs*self.step + self.T)]
            y[bs, 0] = trainy[bs * self.step + self.T + 1] # force
            yd[bs,0] = abs(trainy[bs*self.step + self.T +1] - trainy[bs*self.step + self.T -50])*10+0.05 # force derative
        yd[:,0] = [abs(item) / 5 if abs(item) > 0.2 else abs(item) for item in yd[:,0]]
        #return np.squeeze(x, axis=0), np.squeeze(y, axis=0) # 0 dimm is batch for dataloader, but not the real batch.
        #return x.astype(np.float64),y.astype(np.float64) # x shape: ([1, 28, 1, 110, 1000]), y shape: ([1, 28, 1])
        return x, y

    def __len__(self):
        return self.y.shape[0]

class SEEGDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        assert self.x.shape[0] == self.y.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.y.shape[0]

def read_rawdata():
    datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move1TrainRawData.mat'
    datafile11 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move1TestRawData.mat'
    datafile2 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move2TrainRawData.mat'
    datafile21 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move2TestRawData.mat'
    datafile3 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move3TrainRawData.mat'
    datafile31 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move3TestRawData.mat'
    datafile4 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TrainRawData.mat'
    datafile41 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TestRawData.mat'
    data1 = scipy.io.loadmat(datafile1)
    data11 = scipy.io.loadmat(datafile11)
    data2 = scipy.io.loadmat(datafile2)
    data21 = scipy.io.loadmat(datafile21)
    data3 = scipy.io.loadmat(datafile3)
    data31 = scipy.io.loadmat(datafile31)
    data4 = scipy.io.loadmat(datafile4)
    data41 = scipy.io.loadmat(datafile41)

    traintmp1 = data1['train']
    traintmp11= data11['test']
    traintmp2 = data2['train']
    traintmp21 = data21['test']
    traintmp3 = data3['train']
    traintmp31 = data31['test']
    traintmp4 = data4['train']
    traintmp41 = data41['test']

    dataset=np.concatenate((traintmp1,traintmp11,traintmp2,traintmp21,traintmp3,traintmp31,traintmp4,traintmp41),axis=0) # (2010000, 20)

    # fs=1000, get 5 test period.
    testx=np.transpose(dataset[-75000:,:-1]) #(114, 1500)
    testy=np.transpose(dataset[-75000:,-1]) # (1500,)
    trainx=np.transpose(dataset[0:-75000,:-1]) #  (114, 38692)
    trainy=np.transpose(dataset[:-75000,-1]) # (38692,)

    del data1,data11,data2,data21,data3,data31,data4,data41
    del traintmp1, traintmp11,traintmp2,traintmp21,traintmp3,traintmp31,traintmp4,traintmp41

    return trainx,trainy,testx, testy


def read_fbanddata():
    # input is window smoothed: f=1/50ms
    datafile1 = '/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/move1TrainData3D.mat'
    datafile2 = '/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/move2TrainData3D.mat'
    datafile3 = '/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/move3TrainData3D.mat'
    datafile4 = '/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/move4TrainData3D.mat'
    data1 = scipy.io.loadmat(datafile1)
    data2 = scipy.io.loadmat(datafile2)
    data3 = scipy.io.loadmat(datafile3)
    data4 = scipy.io.loadmat(datafile4)

    traintmp1 = data1['train'] #(182, 299, 36)
    traintmp2 = data2['train'] #(182, 299, 37)
    traintmp3 = data3['train'] #(182, 299, 30)
    traintmp4 = data4['train'] #(182, 299, 38)

    testNum=2
    train=np.concatenate((traintmp1[:,:,:-testNum],traintmp2[:,:,:-testNum],traintmp3[:,:,:-testNum],traintmp4[:,:,:-testNum]),axis=2) # (182, 299, 98)
    test = np.concatenate((traintmp1[:, :, -testNum:],traintmp2[:, :, -testNum:],traintmp3[:, :, -testNum:], traintmp4[:, :, -testNum:]), axis=2) #(182, 299, 6)

    trainx=train[:-2,:,:] #(180, 299, 133)
    trainy=train[180,:,:] #(299, 133)
    testx=test[:-2,:,:] #(180, 299, 8)
    testy=test[180,:,:] #(299, 8)

    return trainx,trainy,testx, testy

def plotloss(ax,trainlose,testlost):
    with open(trainlose) as f:
        trainlose = f.read().splitlines()
    with open(testlost) as f:
        testlose = f.read().splitlines()
    x=len(testlose)
    trainlose=[float(x) for x in trainlose]
    testlose = [float(x) for x in testlose]
    plt.cla()
    ax.plot(testlose, label='test lose')
    ax.plot(trainlose, label='train lose')
    plt.legend()
    plt.show()
    plt.pause(2)
    plt.savefig('trainAndTestLoseCurve.png')

# system will hang if read all four movement raw data as one 3D.
#TODO: not working, data too big when depth=500, system hang
def read_rawdataAs3D(depth=500,prediction=1):
    datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move1TrainRawData.mat'
    datafile11 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move1TestRawData.mat'
    datafile2 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move2TrainRawData.mat'
    datafile21 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move2TestRawData.mat'
    datafile3 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move3TrainRawData.mat'
    datafile31 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move3TestRawData.mat'
    datafile4 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TrainRawData.mat'
    datafile41 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TestRawData.mat'
    data1 = scipy.io.loadmat(datafile1)
    data11 = scipy.io.loadmat(datafile11)
    data2 = scipy.io.loadmat(datafile2)
    data21 = scipy.io.loadmat(datafile21)
    data3 = scipy.io.loadmat(datafile3)
    data31 = scipy.io.loadmat(datafile31)
    data4 = scipy.io.loadmat(datafile4)
    data41 = scipy.io.loadmat(datafile41)

    traintmp1 = data1['train']
    traintmp11= data11['test']
    traintmp2 = data2['train']
    traintmp21 = data21['test']
    traintmp3 = data3['train']
    traintmp31 = data31['test']
    traintmp4 = data4['train']
    traintmp41 = data41['test']
    #dataset=np.concatenate((traintmp1,traintmp11,traintmp2,traintmp21,traintmp3,traintmp31,traintmp4,traintmp41),axis=0).transpose() # (2010000, 20)
    # still too big:  print("%d bytes" % (ds1.size * ds1.itemsize))
    ds1 = np.concatenate((traintmp1, traintmp11),axis=0).transpose()
    ds2 = np.concatenate((traintmp2, traintmp21), axis=0).transpose()
    ds3 = np.concatenate((traintmp3, traintmp31), axis=0).transpose()
    ds4 = np.concatenate((traintmp4, traintmp41), axis=0).transpose()

    trainx1 = np.zeros((depth,ds1.shape[0]-1,ds1.shape[1]-depth+1)) # (depth, feature, time)
    trainy1 = np.zeros((ds1.shape[1], 1))

    tmpdataset = ds1[:19, :]
    for i in range(depth):
        trainx1[i,:,:]=tmpdataset[:,i:-(depth-i-1)]
    pass
    #return trainx,trainy,testx, testy

def rawData(split=True,move2=True): # del
    activeChannels = [8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 62, 63, 69, 70, 105, 107, 108, 109, 110,111] # 111 is force channel
    badtrials={}

    badtrials['1'] = np.array(badtrial1) - 1
    badtrials['2'] = np.array(badtrial2) - 1
    badtrials['3'] = np.array(badtrial3) - 1
    badtrials['4'] = np.array(badtrial4) - 1

    activeChannels=np.array(activeChannels)-1
    datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move1.mat'
    datafile2 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move2.mat'
    datafile3 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move3.mat'
    datafile4 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move4.mat'
    data1 = scipy.io.loadmat(datafile1)
    data2 = scipy.io.loadmat(datafile2)
    data3 = scipy.io.loadmat(datafile3)
    data4 = scipy.io.loadmat(datafile4)
    data1 = data1['data'] # (channel, time, trials)
    data2 = data2['data']
    data3 = data3['data']
    data4 = data4['data']
    data = {}
    data['1'] = np.float32(data1)
    data['2'] = np.float32(data2)
    data['3'] = np.float32(data3)
    data['4'] = np.float32(data4)


    for i in range(4):
        move=str(int(float(i)) + 1)
        alltrialidx = range(data[move].shape[2])  # 0--39
        trialidx = np.setdiff1d(alltrialidx, badtrials[str(int(float(i)) + 1)])
        data[move] = data[move][activeChannels, :, :]
        data[move] = data[move][:, :, trialidx]  # (channels, time,trials), (20, 15000, 33) # last channel is force

    if split==True:
        traindatatmp={}
        valdatatmp={}
        testdatatmp = {}
        testNum = 2 # 2*4=8 test trials
        valNum = 2 # 2*4=8 valuate trials
        for i in range(4):
            move = str(int(float(i)) + 1)
            valdatatmp[move] = data[move][:,:,-(valNum):] # including -1(last) and -2. (20, 15000, 2)
        for i in range(4):
            move = str(int(float(i)) + 1)
            testdatatmp[move] = data[move][:, :, -(testNum+valNum):-(valNum)]  # including -4:-2,  (20, 15000, 2)
        for i in range(4):
            move = str(int(float(i)) + 1)
            traindatatmp[move] = data[move][:, :, :-(testNum+valNum)]

        if move2 == True:
            valdata=np.concatenate((valdatatmp['1'],valdatatmp['2'],valdatatmp['3'],valdatatmp['4'])
                                   ,axis=2) #(20, 15000, 8)
            testdata = np.concatenate((testdatatmp['1'], testdatatmp['2'], testdatatmp['3'], testdatatmp['4'])
                                      ,axis=2) #(20, 15000, 8)
            traindata = np.concatenate((traindatatmp['1'], traindatatmp['2'], traindatatmp['3'], traindatatmp['4'])
                                       ,axis=2) #(20, 15000, 118)
        else:
            valdata = np.concatenate((valdatatmp['1'], valdatatmp['3'], valdatatmp['4']),
                                     axis=2)  # (20, 15000, 8)
            testdata = np.concatenate((testdatatmp['1'], testdatatmp['3'], testdatatmp['4']),
                                      axis=2)  # (20, 15000, 8)
            traindata = np.concatenate((traindatatmp['1'], traindatatmp['3'], traindatatmp['4']),
                                       axis=2)  #
        return traindata, valdata, testdata
    return data #  #


# convert 2D signle trial data in channelx*times to channels*windSize*steps
# input traildata contains seeg data and 1 channel of force data
def windTo3D(trialdata,wind,step=1):
    trainy=trialdata[-1,:] # force
    trainy=trainy[wind:]

    channs=trialdata.shape[0] -1
    T = trialdata.shape[1] # T: 15000
    slides = T - wind  # 14500
    trainx = np.zeros((channs, slides, wind))  # (19, 14500, 500)

    for i in range(wind):
        # print(i)
        if i == (wind - 1):
            trainx[:, :, i] = trialdata[:, i:-1]
        else:
            trainx[:, :, i] = trialdata[:, i:-(wind - i)]

    yield trainx, trainy
def rawData2(rawOrBand,activeChannels,split=True,move2=True):
    #call: rawData2('raw'/'band',[list]/'all',split=True,move2=True)
    #basedir='/Users/long/BCI/python_scripts/grasp/process/'
    #from grasp.config import raw_data, activeChannels
    #activeChannels=[8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 62, 63, 69, 70, 105, 107, 108, 109, 110]
    movements=4

    if rawOrBand=='raw':
        file_suffix='epoch.fif'
    elif rawOrBand=='band':
        file_suffix = 'BandEpoch.fif'
    moves=[]
    for i in range(movements):
        moves.append([])
        # ignore the stim channel
        moves[i]=mne.read_epochs(data_dir+ 'move'+str(i)+file_suffix).get_data(picks=['seeg', 'emg']).transpose(1,2,0)

    # take target force only
    if isinstance(activeChannels,list):
        activeChannels = activeChannels + [-2, ]  # realforce+target +stim(ignored above)
    else:
        activeChannels=range(moves[0].shape[0])

    if move2==True:
        allmove=[0,1,2,3]
    else:
        allmove=[0,2,3]
    for i in allmove:
        #movecode=str(int(float(i)) + 1)
        alltrialidx = range(moves[i].shape[2])  # 0--39
        trialidx = np.setdiff1d(alltrialidx, badtrials[i])
        moves[i] = moves[i][activeChannels, :, :]
        moves[i] = moves[i][:, :, trialidx]  # (channels, time,trials), (20=19+1/116=114+2, 15000, 33) # last channel is force
    if split==True:
        traindatatmp=[]
        valdatatmp=[]
        testdatatmp = []
        testNum = 2 # 2*4=8 test trials
        valNum = 2 # 2*4=8 valuate trials
        for i in allmove:
            valdatatmp.append([])
            valdatatmp[i] = moves[i][:,:,-(valNum):] # including -1(last) and -2. (20, 15000, 2)
        for i in allmove:
            testdatatmp.append([])
            testdatatmp[i] = moves[i][:, :, -(testNum+valNum):-(valNum)]  # including -4:-2,  (20, 15000, 2)
        for i in allmove:
            traindatatmp.append([])
            traindatatmp[i] = moves[i][:, :, :-(testNum+valNum)]
        valdata=np.concatenate((valdatatmp),axis=2) #(20, 15000, 8)
        testdata = np.concatenate((testdatatmp),axis=2) #(20, 15000, 8)
        traindata = np.concatenate((traindatatmp),axis=2) #(20, 15000, 118)
        return traindata, valdata, testdata
    return moves

#import grasp.utils as utils
#aa=utils.rawData()
#np.save('/Users/long/Documents/BCI/python_scripts/grasp/data/rawdata.npy', aa)

# a smaller set, few trials
def rawDataSmaller(trialnum=30):
    wholedata=np.load('/Users/long/Documents/BCI/python_scripts/grasp/data/rawdata.npy',allow_pickle='TRUE').item()
    wholedata1=np.concatenate((wholedata['1'],wholedata['2'],wholedata['3']),axis=2) # 3 movement data (20, 15000, 99)
    idx=list(range(99))
    randtrial=random.sample(idx,trialnum)
    left = list(np.setdiff1d(idx, randtrial))
    pickone=random.sample(left,1)
    train=wholedata1[:,:,randtrial] #(20, 15000, 30)
    test=wholedata1[:,:,pickone] #(20, 15000, 2)
    return train,test

# save each 15s movement as a 3D cube to disk
def preprocess():
    data=rawData()
    chns = 19
    depth = 500  # 500 ms
    predict = 1  # predict next point
    print('lala')
    for k, (movetype,movedata) in enumerate(data.items()):
        if k==0:
            # iterate every single trial
            triallen = movedata.shape[1]
            trialnum=movedata.shape[2]
            len = triallen - depth  # 14500
            trainx = np.zeros((chns, len, depth))  # (19, 14500, 500)
            # put all trials force in one array
            trainy=np.zeros((len,trialnum.shape[0])) # len * trialNumber

            for trial in range(trialnum):
                print('processing ' + str(movetype) + ' , trial:' + str(trial + 1))
                force=movedata[-1,:,trial] # foce data
                trialdata=movedata[:-1,:,trial] #(15000, 19)
                trainy[:,trial]=force[500:] # (14500,)
                for i in range(depth):
                    #print(i)
                    if i==(depth-1):
                        trainx[:, :, i] = trialdata[:, i:-1]
                    else:
                        trainx[:,:,i] = trialdata[:, i:-(depth - i)]
                filename='/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/dataloader/move'+str(movetype)+'Trial'+str(trial)
                np.save(filename, trainx)
            filename = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/dataloader/force' + str(movetype)
            print('save force data.')
            np.save(filename, trainy.transpose())
#forcefile='/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/dataloader/train/force1.npy'
#force=np.load(forcefile)

# iter 3D 15s data from disk
class itermove(Dataset):
    def __init__(self, root_dir='/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/dataloader/train', movetype=1):
        self.root_dir = root_dir
        self.movetype = movetype
        self.force=np.load(os.path.join(self.root_dir,'force'+str(movetype)+'.npy')) # (time14500,trials36)

    def __len__(self):
        return self.force.shape[1]

    def __getitem__(self, idx):
        movedatafile = os.path.join(self.root_dir,'move'+str(self.movetype)+'Trial'+str(idx)+'.npy')
        movedata=np.load(movedatafile)
        force = self.force[:,idx]
        data = { 'data': movedata, 'force': force}
        return data


def read_rawdata_3D(fullOrTruncate,realOrTarget):
    activeChannels = np.asarray([8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 62, 63, 69, 70, 105, 107, 108, 109, 110])-1
    alltrials=np.arange(40)
    badtrial1=np.asarray([22, 23, 24, 26]);  badtrial1=badtrial1-1
    badtrial2= np.asarray([23, 24, 28]); badtrial2=badtrial2-1
    badtrial3= np.asarray([4, 8, 11, 16, 21,22,23,24,25, 29]); badtrial3=badtrial3-1
    badtrial4= np.asarray([24, 29]); badtrial4=badtrial4-1
    goodtrial1 = np.setdiff1d(alltrials, badtrial1)
    goodtrial2 = np.setdiff1d(alltrials, badtrial2)
    goodtrial3 = np.setdiff1d(alltrials, badtrial3)
    goodtrial4 = np.setdiff1d(alltrials, badtrial4)

    datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move1.mat'
    datafile2 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move2.mat'
    datafile3 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move3.mat'
    datafile4 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move4.mat'

    data1 = scipy.io.loadmat(datafile1)
    data2 = scipy.io.loadmat(datafile2)
    data3 = scipy.io.loadmat(datafile3)
    data4 = scipy.io.loadmat(datafile4)

    #(112, 15000, 40)
    datatmp1 = data1['data'][:,:,goodtrial1] #(113, 15000, 36)
    datatmp2 = data2['data'][:,:,goodtrial2] #(113, 15000, 37)
    datatmp3 = data3['data'][:,:,goodtrial3] #(113, 15000, 30)
    datatmp4 = data4['data'][:,:,goodtrial4] #(113, 15000, 38)
    dataset=np.concatenate((datatmp1,datatmp2,datatmp3,datatmp4),axis=2) # (112, 15000, 104)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    #for trial in range(dataset.shape[2]):
    #    dataset[:,:,trial] = np.transpose(scaler.fit_transform(np.transpose(dataset[:,:,trial])))

    if fullOrTruncate=='truncate':
        dataset=dataset[:,0:10000,:] # only regress to early 10s
    elif fullOrTruncate=='full':
        pass
    else:
        print('full or truncate')
        return 1
    del data1,data3,data4,datatmp1,datatmp3,datatmp4
    index=np.arange(dataset.shape[2])
    picktest=[20,21,40,41,80,81,120,121]
    picktrain = np.setdiff1d(index, picktest)

    if realOrTarget=='real':
        extraChannel=110
    elif realOrTarget=='target':
        extraChannel = 111
    else:
        print ('Predict on which signal: real or target')
        return 1
    channels=np.concatenate((activeChannels,np.asarray([extraChannel,])),axis=0) # 110 is real forct, 111 is target
    trainx = dataset[channels,:,:][:,:,picktrain] #(20, 15000, 94)
    trainy = trainx[-1,:,:] #(15000, 94)
    #trainy=np.reshape(trainy,[1,-1],order='F'), then plot to check data
    trainx=trainx[:-1,:,:] #(19, 15000, 94)

    testx = dataset[channels, :, :][:,:,picktest]  # (109, 15000, 20)
    testy = testx[-1, :, :]  # (15000, 20)
    # testy=np.reshape(testy,[1,-1],order='F'), plt.plot(np.squeeze(testy))
    testx = testx[:-1, :, :]  # (19, 15000, 10)

    return trainx,trainy,testx, testy


# generate 3D data of one trial data on the fly
def gen3DOnTheFly(traindata): # traindata: (20channels, 15000times, trialNum)
    chns = 19
    wind=500
    #for k, (movetype,movedata) in enumerate(traindata.items()):
    trialnum = traindata.shape[2]
    for trial in range(trialnum): # for each single 15s movedata
        tmp=traindata[:,:,trial] # trialdata: 20*15000, movedata:channel*time*trials
        #trialdata=tmp[:-1,:] # 19*15000
        trialdata = tmp[:, :]  # 20*15000  # use history force data
        trainy=tmp[-1,:] # force
        trainy=trainy[wind:]

        T = traindata.shape[1] # T: 15000
        len = T - wind  # 14500
        trainx = np.zeros((chns+1, len, wind))  # (19, 14500, 500)

        for i in range(wind):
            # print(i)
            if i == (wind - 1):
                trainx[:, :, i] = trialdata[:, i:-1]
            else:
                trainx[:, :, i] = trialdata[:, i:-(wind - i)]

        yield trainx, trainy


def readRestForModelTest():
    activeChannels = np.asarray([8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 62, 63, 69, 70, 105, 107, 108, 109, 110])-1
    alltrials=np.arange(40)
    testtrial1=np.asarray([24, 26]);  testtrial1=testtrial1-1
    testtrial2= np.asarray([23, 24]); testtrial2=testtrial2-1
    testtrial3= np.asarray([8,21,29]); testtrial3=testtrial3-1
    testtrial4= np.asarray([24,]); testtrial4=testtrial4-1

    datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move1.mat'
    #datafile2 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move2.mat'
    datafile3 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move3.mat'
    datafile4 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move4.mat'

    data1 = scipy.io.loadmat(datafile1)
    #data2 = scipy.io.loadmat(datafile2)
    data3 = scipy.io.loadmat(datafile3)
    data4 = scipy.io.loadmat(datafile4)

    #(112, 15000, 40)
    datatmp1 = data1['data'][:,:,testtrial1] #(113, 15000, 36)
    #datatmp2 = data2['data'][:,:,goodtrial2] #(113, 15000, 37)
    datatmp3 = data3['data'][:,:,testtrial3] #(113, 15000, 30)
    datatmp4 = data4['data'][:,:,testtrial4] #(113, 15000, 38)
    dataset=np.concatenate((datatmp1,datatmp3,datatmp4),axis=2) # (112, 15000, 104)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    #for trial in range(dataset.shape[2]):
    #    dataset[:,:,trial] = np.transpose(scaler.fit_transform(np.transpose(dataset[:,:,trial])))

    dataset=dataset[:,0:10000,:] # only regress to early 10s
    del data1,data3,data4,datatmp1,datatmp3,datatmp4

    channels=np.concatenate((activeChannels,np.asarray([111,])),axis=0) # 110 is real forct, 111 is target
    testx = dataset[channels,:,:] #(20, 15000, 94)
    testy = testx[-1,:,:] #(15000, 94)
    #trainy=np.reshape(trainy,[1,-1],order='F'), then plot to check data
    testx=testx[:-1,:,:] #(19, 15000, 94)

    return testx,testy

def ukfInput(trainNum=10):
    # trainNum = 10
    # 30 active channels
    datafile1 = '/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/move1TrainData3D.mat'
    datafile2 = '/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/move2TrainData3D.mat'
    datafile3 = '/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/move3TrainData3D.mat'
    datafile4 = '/Users/long/BCI/matlab_scripts/force/data/SEEG_Data/move4TrainData3D.mat'
    data1 = scipy.io.loadmat(datafile1)
    data2 = scipy.io.loadmat(datafile2)
    data3 = scipy.io.loadmat(datafile3)
    data4 = scipy.io.loadmat(datafile4)

    traintmp1 = data1['train']  # (182, 299, 36)
    traintmp2 = data2['train']  # (182, 299, 37)
    traintmp3 = data3['train']  # (182, 299, 30)
    traintmp4 = data4['train']  # (182, 299, 38)

    # total train number: 10+10+10+10=40
    train = np.concatenate((traintmp1[:, :, :trainNum], traintmp2[:, :, :trainNum], traintmp3[:, :, :trainNum], traintmp4[:, :, :trainNum]),axis=2)  # (182, 299, 98)
    # rest are testing trials
    test = np.concatenate((traintmp1[:, :, trainNum:], traintmp2[:, :, trainNum:], traintmp3[:, :, trainNum:], traintmp4[:, :, trainNum:]),axis=2)  # (182, 299, 6)

    trainx = train[:-2, :, :]  # (180, 299, 30)
    trainx = trainx.swapaxes(1, 2)
    trainx = trainx.swapaxes(0, 1)
    trainx = trainx.swapaxes(1, 2) # (30, 299, 180)
    trainy = train[-2, :, :]  # (299, 30)

    testx = test[:-2, :, :]  # (180, 299, 74)
    testx = testx.swapaxes(1, 2)
    testx = testx.swapaxes(0, 1)  # (74, 180, 299)
    testx = testx.swapaxes(1, 2)  # (74, 299, 180)
    testy = test[-2, :, :]  # (299, 74)

    dt=0.05 #ms
    # Y should start from the second time point, discard 0 and 1.
    trainy_tmp=trainy.T # (30, 299)
    diff1=trainy_tmp[:,1:]-trainy_tmp[:,:-1] # (30, 298)
    rateTrain=diff1[:,1:] # (30, 297)
    diff2=diff1[:,1:]-diff1[:,:-1] # (30, 297)
    yankTrain=diff2

    trainy=np.zeros((yankTrain.shape[0],yankTrain.shape[1],3)) # (30 trials, 297 time points, 3 elements), 3 element: force, rate, yank
    trainy[:,:,0] = trainy_tmp[:, 2:]  # (30, 297)
    trainy[:,:,1] = rateTrain/dt
    trainy[:,:,2] = yankTrain/dt
    trainy=trainy

    # Y should start from the second time point
    #TODO: plot show yank provide little info, much like force rate
    testy_tmp = testy.T  # (30, 299)
    diff11 = testy_tmp[:, 1:] - testy_tmp[:, :-1]  # (30, 298)
    rateTest = diff11[:, 1:]  # (30, 297)
    diff22 = diff11[:, 1:] - diff11[:, :-1]  # (30, 297)
    yankTest = diff22

    testy = np.zeros((yankTest.shape[0], yankTest.shape[1],3))  # (30 trials, 297 time points, 3 elements), 3 element: force, rate, yank
    testy[:, :, 0] = testy_tmp[:, 2:]  # (30, 297)
    testy[:, :, 1] = rateTest/dt
    testy[:, :, 2] = yankTest/dt

    return trainx, trainy, testx, testy