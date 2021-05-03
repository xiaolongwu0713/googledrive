import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

from sklearn.decomposition import PCA
import time

from grasp.process.signalProcessUtils import butter_lowpass_filter
from grasp.utils import freq_input
import math
from scipy import signal
from sklearn.cross_decomposition import PLSRegression


sid=6
data=freq_input(sid,split=False,move2=True)
train_test_split=0.7 # train_trials/test_trials = 0.5

# extract some trials as test from all 4 moves
train_data=[]
test_data=[]
for move in range(len(data)):
    trials = data[move].shape[2]
    train_trials=math.floor(trials*train_test_split)
    train_data.append(data[move][:,:,:train_trials])
    test_data.append(data[move][:, :, train_trials:])
train_data=np.concatenate(train_data,axis=2) # (116, 15001, 92)
test_data=np.concatenate(test_data,axis=2) # (116, 15001, 42)

# shuffle
train_data=train_data.transpose(2,0,1) # (trail,channels,times)
test_data=test_data.transpose(2,0,1)
np.random.shuffle(train_data)
np.random.shuffle(test_data)

# flat trials in 3D into 2D
train_data=train_data.transpose(1,0,2) # (channels,trail,times)
test_data=test_data.transpose(1,0,2)
train_data=np.reshape(train_data,(train_data.shape[0],-1))
test_data=np.reshape(test_data,(test_data.shape[0],-1))
train_x=train_data[:-2,:] # (114, 1380092)
train_y=train_data[-2,:]
test_x=test_data[:-2,:]
test_y=test_data[-2,:]

# down sample to 1000/5=200HZ
train_data = signal.decimate(train_data, 5, axis=1,ftype='iir',zero_phase=True)  # (116, 276019)
test_data = signal.decimate(test_data, 5, axis=1,ftype='iir',zero_phase=True)
train_x=train_data[:-2,:] # (114, 1380092)
train_y=train_data[-2,:]
test_x=test_data[:-2,:]
test_y=test_data[-2,:]

pls = PLSRegression(n_components=10)
pls.fit(train_x.T, train_y) # input: (n_samples, n_features)
a=pls.predict(test_x.T)

# very noisy
plt.plot(a[:,0],color='green',linewidth=1)
plt.plot(test_y,color='red',linewidth=1)

fs=200
cutoff=1
b=butter_lowpass_filter(np.squeeze(a),cutoff, fs, order=5)

# Looks better after lowpass
plt.plot(b,color='green',linewidth=1)
plt.plot(test_y,color='red',linewidth=1)