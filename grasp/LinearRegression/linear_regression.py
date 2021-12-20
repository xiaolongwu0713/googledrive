'''
Linear regression.
1, linear regression
2, polynomial regression
Note: nn.MSE_loss() will do average itself.
3, It will produce mse of individual trial for all 4 movements, so statistic analysis can be done.
Try to use SPoC: https://mne.tools/stable/auto_examples/decoding/decoding_spoc_CMC.html#sphx-glr-auto-examples-decoding-decoding-spoc-cmc-py
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline

from grasp.config import *
from grasp.process.channel_settings import activeChannels
from grasp.process.signalProcessUtils import butter_lowpass_filter
from grasp.utils import load_data
import math
from scipy import signal
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

sid=16

print('Subject ID: '+ str(sid)+ '.')
plot_dir=data_dir + 'PF' + str(sid) +'/prediction/linear/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

polynomial=False # polynomial fitting
input_freq_for_linear_reg=False # True: frequency + raw. False: Raw input.
standardization=True
print('Polynomial: '+ str(polynomial)+'.')

print('Reading data....')
# freq_input has too many feature. Memory won't fit after features are raised to power.
if polynomial==True: # polynomial+raw
    data = load_data(sid,split=False,move2=True,input='raw')
if polynomial==False and input_freq_for_linear_reg==True: # linear fitting on freq+raw
    data = load_data(sid, split=False, move2=True,input='frequency_and_raw')
elif polynomial==False and input_freq_for_linear_reg==False: # linear fitting on raw
    data = load_data(sid, split=False, move2=True,input='raw')


train_test_split=0.7 # train_trials/test_trials = 0.5

# extract some trials as test from all 4 moves
print('Train test data split: '+str(train_test_split)+'.')
train_data=[]
test_data=[]
train_trials_num=[]
test_trials_num=[]
testNum=8
for move in range(len(data)):
    trials = data[move].shape[2]
    #train_trials=math.floor(trials*train_test_split) # or use a fixed test trial
    train_trials=trials-testNum
    train_trials_num.append(train_trials)
    test_trials_num.append(trials-train_trials)
    train_data.append(data[move][:,:,:train_trials])
    test_data.append(data[move][:, :, train_trials:])
train_data=np.concatenate(train_data,axis=2) # (116, 15001, 92)
test_data=np.concatenate(test_data,axis=2) # (116, 15001, 42)
print('Train data shape(channels,time,trials):'+str(train_data.shape))
print('Test data shape(channels,time,trials):'+str(test_data.shape))
del data
# shuffle
print('Shuffle training dataset.')
train_data=train_data.transpose(2,0,1) # (trail,channels,times)
test_data=test_data.transpose(2,0,1)
np.random.shuffle(train_data)
#np.random.shuffle(test_data)

# flat trials in 3D into 2D
print('Flatten 3D to 2D.')
test_data_stat=test_data.copy()
train_data=train_data.transpose(1,0,2) # (channels,trail,times)
test_data=test_data.transpose(1,0,2)
train_data=np.reshape(train_data,(train_data.shape[0],-1))
test_data=np.reshape(test_data,(test_data.shape[0],-1))

# down sample to 1000/5=200HZ
down_sample_factor=5
new_frequency=1000/down_sample_factor
print('Down sample data to ' + str(new_frequency)+'HZ.')
train_data = signal.decimate(train_data, down_sample_factor, axis=1,ftype='iir',zero_phase=True)  # (116, 276019)
test_data = signal.decimate(test_data, down_sample_factor, axis=1,ftype='iir',zero_phase=True)
test_data_tmp=[]
for i in range(test_data_stat.shape[0]):
    test_data_tmp.append([])
    test_data_tmp[i]=signal.decimate(test_data_stat[i,:,:], down_sample_factor, axis=1,ftype='iir',zero_phase=True)
del test_data_stat
test_data_stat=np.asarray(test_data_tmp)


train_x=np.transpose(train_data[:-2,:]) # (114, 1380092)-->(1380092, 114)
train_y=np.squeeze(train_data[-2,:])
test_x=np.transpose(test_data[:-2,:])
test_y=np.squeeze(test_data[-2,:])
test_stat_x=test_data_stat[:,:-2,:]
test_stat_y=np.squeeze(test_data_stat[:,-2,:])

if standardization==True:
    print('Standardization testing and training set. ')
    scaler = StandardScaler(copy=False) # do inplace scaling
    scaler.fit(train_x)
    scaler.transform(train_x)
    scaler.fit(test_x)
    scaler.transform(test_x)
    for i in range(test_stat_x.shape[0]):
        test_stat_x[i] = np.transpose(scaler.transform(np.transpose(test_stat_x[i]),copy=True))

linear_reg=LinearRegression()
criterion = nn.MSELoss() # reduction='sum'
if polynomial==True:
    poly_features=PolynomialFeatures(degree=2)
    lr_model = Pipeline(steps=[('poly_feature', poly_features), ('regressor', linear_reg)])
else:
    lr_model=linear_reg

print('Fitting.')
lr_model.fit(train_x, train_y)
print('Predict on test dataset.')
pred_tmp=lr_model.predict(test_x)
mse_loss=criterion(torch.from_numpy(pred_tmp),torch.from_numpy(test_y.copy()))
## BUG
b=test_y/(1.75/0.7)
b_mean=np.ones((len(b),))*b.mean() # b.mean()=0.17
change_loss=criterion(torch.from_numpy(b),torch.from_numpy(b_mean)) # 0.04
# scale down will decrease MSE
mse_loss2=criterion(torch.from_numpy(pred_tmp/(1.75/0.7)),torch.from_numpy(test_y.copy()/(1.75/0.7))) # 0.03

#avg_loss=a_loss/sum(test_trials_num)
#norm_loss=avg_loss*(default_frequency/new_frequency)
print('MSE loss for SID'+str(sid)+' is: '+str(np.round(mse_loss.numpy(),4)))
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
ax.text(0.45,0.9,'MSE: '+str(np.round(mse_loss.numpy(),4)),fontsize=10,transform=fig.transFigure)
if polynomial==True:
    figname = plot_dir+'polynomial_linear_regression.pdf'
if polynomial==False and input_freq_for_linear_reg == True:
    figname = plot_dir+'linear_regression_on_freq.pdf'
elif polynomial==False and input_freq_for_linear_reg == False:
    figname = plot_dir + 'linear_regression_on_raw.pdf'

fig.savefig(figname)
plt.close(fig)

mse=[]
for i in range(test_stat_x.shape[0]):
    mse.append([])
    pred_tmp=lr_model.predict(np.transpose(test_stat_x[i])) # (126009, 114)
    mse[i]=criterion(torch.from_numpy(pred_tmp),torch.from_numpy(test_stat_y[i].copy())).cpu().detach().item()

filename= plot_dir + 'mse_loss_'+str(testNum*4)+'trials'
np.save(filename,mse)
#trainx=np.load('dir/dir/trainx.npy')