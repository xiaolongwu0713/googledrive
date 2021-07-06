import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from grasp.config import data_dir
from grasp.process.signalProcessUtils import butter_lowpass_filter
from grasp.utils import load_data
import math
from scipy import signal
from sklearn.cross_decomposition import PLSRegression


sid=6
print('Subject ID: '+ str(sid)+ '.')
plot_dir=data_dir + 'PF' + str(sid) +'/pls/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

print('Read data....')
data=load_data(sid,split=False,move2=True)
train_test_split=0.7 # train_trials/test_trials = 0.5

# extract some trials as test from all 4 moves
print('Train test data split: '+str(train_test_split)+'.')
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
print('Shuffle testing dataset.')
train_data=train_data.transpose(2,0,1) # (trail,channels,times)
test_data=test_data.transpose(2,0,1)
np.random.shuffle(train_data)
np.random.shuffle(test_data)

# flat trials in 3D into 2D
print('Flatten 3D to 2D.')
train_data=train_data.transpose(1,0,2) # (channels,trail,times)
test_data=test_data.transpose(1,0,2)
train_data=np.reshape(train_data,(train_data.shape[0],-1))
test_data=np.reshape(test_data,(test_data.shape[0],-1))
#train_x=train_data[:-2,:] # (114, 1380092)
#train_y=train_data[-2,:]
#test_x=test_data[:-2,:]
#test_y=test_data[-2,:]

# down sample to 1000/5=200HZ
down_sample_factor=5
new_frequency=1000/down_sample_factor
print('Down sample data to ' + str(new_frequency)+'HZ.')
train_data = signal.decimate(train_data, down_sample_factor, axis=1,ftype='iir',zero_phase=True)  # (116, 276019)
test_data = signal.decimate(test_data, down_sample_factor, axis=1,ftype='iir',zero_phase=True)
train_x=train_data[:-2,:] # (114, 1380092)
train_y=train_data[-2,:]
test_x=test_data[:-2,:]
test_y=test_data[-2,:]

# PLS fitting
pls_components=10
print('PLS fitting with '+str(pls_components)+' components.')
pls = PLSRegression(n_components=pls_components)
pls.fit(train_x.T, train_y) # input: (n_samples, n_features)

print('Predict on test dataset.')
pred_tmp=pls.predict(test_x.T)


# very noisy
#plt.plot(a[:,0],color='green',linewidth=1)
#plt.plot(test_y,color='red',linewidth=1)
cutoff=1
print('Denoise the prediction by lowpass prediction to '+ str(cutoff)+'HZ.')
pred=butter_lowpass_filter(np.squeeze(pred_tmp),cutoff, new_frequency, order=5)

print('Output predicting plot to '+plot_dir+' .')
fig,ax=plt.subplots()
# Looks better after lowpass
ax.plot(pred,color='green',linewidth=0.3)
ax.plot(test_y,color='red',linewidth=0.3)

figname = plot_dir+'pls_regression.pdf'
fig.savefig(figname)
plt.close(fig)