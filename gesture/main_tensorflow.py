#%cd /content/drive/MyDrive/
# raw_data is imported from global config

#%%capture
#! pip install mne==0.19.2;
#! pip install torch;
#! pip install tensorflow-gpu == 1.12.0;#however program runs fine under tensorflow==2.3.1 on my Mac.

import numpy as np
from scipy.io import savemat
from sklearn.model_selection import StratifiedKFold
import os
import h5py
from gesture.models.EEGModels import DeepConvNet_210519_512_10, EEGNet, EEGNet_lsj
from gesture.config import root_dir, data_dir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


classNum = 5
Inf = [[2, 1000], [3, 1000], [ 4, 1000], [5, 1000], [ 7, 1000], [ 8, 1000], [ 9, 1000], [ 10, 2000],
    [13, 2000], [ 16, 2000], [ 17, 2000], [ 18, 2000], [ 19, 2000], [ 20, 1000], [ 21, 1000], [ 22, 2000], [ 23, 2000],  #[24, 2000], [ 25, 2000], [ 26, 2000],
     [  29, 2000], [ 30, 2000], [ 31, 2000], [ 32, 2000], [ 34, 2000], [ 35, 1000],
    [36, 2000], [ 37, 2000], [41, 2000]]
Inf = np.array(Inf)
#Inf = Inf[[0,1,2,7,8,11,15,17,20,21,25],:]
Inf = Inf[[7],:]

subjNum = np.size(Inf, 0)
kfolds = 2
repeatTs = 1
epochsCt = 50

accuracy = np.zeros(( subjNum, epochsCt ))
loss = np.zeros(( subjNum, epochsCt ))
#for subj in range( subjNum ):
for subj in [4,]:
    sampleRate = 1000
    #pn  = Inf[subj, 0]
    pn=subj

    loadPath = data_dir+'preprocessing/P'+str(pn)+'/preprocessingALL_3_Algorithm_v3.mat'
    #loadPath = 'H:/lsj/preprocessing_data/P' + str(pn) + '/preprocessing3_Algorithm/preprocessingALL_3_Algorithm_v3.mat'
    matDict = h5py.File(loadPath, 'r')
    data = matDict['preData']
    data = np.transpose(data, (4, 3, 2, 1, 0))
    label = matDict['preLabel']
    label = np.transpose(label, (1, 0))
    label = label.astype('int64')
    tampLbl = label[:, 1]
    trial, strideInx, kernel, channel, sample = data.shape
    Inx = np.arange(trial)

    tampCt = 0
    _accuracy = np.zeros((kfolds * repeatTs, epochsCt))
    _loss = np.zeros((kfolds * repeatTs, epochsCt))
    for ex in range(repeatTs):
        tkFold = StratifiedKFold(n_splits = kfolds, shuffle=True)
        for i, j in tkFold.split(Inx, tampLbl):
            tampCt += 1
            trainData, testData = data[i], data[j]
            trainLabel, testLabel = label[i], label[j]
            trainData = np.reshape(trainData, (-1, kernel, channel, sample ))
            testData = np.reshape(testData, (-1, kernel, channel, sample ))
            trainLabel = np.reshape(trainLabel, (-1, 1))
            testLabel = np.reshape(testLabel, (-1, 1))
            trainLabel = np.eye(classNum)[trainLabel - 1]
            testLabel = np.eye(classNum)[testLabel - 1]
            trainLabel = np.reshape(trainLabel, [-1, classNum])
            testLabel = np.reshape(testLabel, [-1, classNum])

            #model = DeepConvNet_210519_512_10(classNum, Chans=channel, Samples=sample,dropoutRate=0.5)
            model = EEGNet_lsj(classNum, Chans=channel, Samples=sample,dropoutRate=0.5)
            model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
            for epoch in range(epochsCt):
                if epoch%5 == 0:
                    print('Epoch: '+str(epoch))
                fitted = model.fit(trainData, trainLabel, batch_size= 32,  epochs= 2)
                los, accu = model.evaluate(testData, testLabel)
                _accuracy[tampCt-1, epoch] = accu
                _loss[tampCt - 1, epoch] = los

    


acc=np.mean(_accuracy,axis=0)

_accuracy.shape, acc.shape

acc


