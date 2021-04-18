"""
Helper functions.

@author Zhenye Na 05/21/2018
@modified 11/05/2019

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).
    [2] Chandler Zuo. "A PyTorch Example to Use RNN for Financial Prediction" (2017).
"""

import numpy as np
import pandas as pd
import scipy.io

def read_data(args, debug=True):

    if args.feature=='fbands':
        train_prototype=args.traindataset
        test_prototype = args.testdataset
        #datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TrainData.mat'
        #datafile2 = '/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TestData.mat'

        # train:(9299, 115), test: (599, 115)
        move1train = scipy.io.loadmat('/Users/long/Documents/BCI/matlab_scripts/force/pls/move1TrainData.mat')['train']
        move1test = scipy.io.loadmat('/Users/long/Documents/BCI/matlab_scripts/force/pls/move1TestData.mat')['test']
        move2train = scipy.io.loadmat('/Users/long/Documents/BCI/matlab_scripts/force/pls/move2TrainData.mat')['train']
        move2test = scipy.io.loadmat('/Users/long/Documents/BCI/matlab_scripts/force/pls/move2TestData.mat')['test']
        move3train = scipy.io.loadmat('/Users/long/Documents/BCI/matlab_scripts/force/pls/move3TrainData.mat')['train']
        move3test = scipy.io.loadmat('/Users/long/Documents/BCI/matlab_scripts/force/pls/move3TestData.mat')['test']
        move4train = scipy.io.loadmat('/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TrainData.mat')['train']
        move4test = scipy.io.loadmat('/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TestData.mat')['test']

        #raw1 = scipy.io.loadmat(datafile1)
        #raw2 = scipy.io.loadmat(datafile2)
        #train = raw1['train']  # (6299, 115)
        #test = raw2['test']  # (2699, 115)
        testNum = 2
        train = np.concatenate((move1train[:, :, :-testNum], traintmp3[:, :, :-testNum], traintmp4[:, :, :-testNum]),
                               axis=2)  # (182, 299, 98)
        test = np.concatenate((traintmp1[:, :, -testNum:], traintmp3[:, :, -testNum:], traintmp4[:, :, -testNum:]),
                              axis=2)  # (182, 299, 6)

        tmp = np.concatenate((move1train, move1test,move2train,move2test,move3train,move3test,move4train,move4test), 0)  # (8998, 115)
        X = tmp[:, 0:-1]  # ([8998, 114])
        y = tmp[:, -1]  # ([8998])
    elif args.feature=='rawmove':
        train_prototype = args.traindataset
        test_prototype = args.testdataset
        datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move' + str(train_prototype) + '.mat'
    elif args.feature=='rawseeg':
        datafile1 = '/Users/long/Documents/BCI/matlab_scripts/force/data/SEEG_Data/move1.mat'
    return X, y
