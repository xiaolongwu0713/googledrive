import math
from numpy.linalg import norm
from math import atan2
import numpy as np
from numpy.random import randn
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
import matplotlib.pyplot as plt
from torch import nn

from grasp.utils import ukfInput
from sklearn.linear_model import Ridge
from grasp.config import *

#Note: Do not run sid=16. Too many feature will hang PC.
sid=6
print('Subject ID: '+ str(sid)+ '.')
plot_dir=data_dir + 'PF' + str(sid) +'/prediction/ukf/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# In Kalman, x mean force info, y mean seeg feature
# trainy/testy: (trials 30 or 74 , time points 299 ,feature 180)
# trainx/testx: (trials 30 or 74,time points 297, type 3) (30, 297, 3)
trainy, trainx, testy, testx = ukfInput(sid)
# testy: (101, 299, 180)-->(180, 30199)
# flatten to 2d
testy=testy.transpose(2,0,1)
testy=np.reshape(testy,(testy.shape[0],-1))

# testx: (101, 297, 3)-->(3, 29997)
# flatten to 2d
testx=testx.transpose(2,0,1)
testx=np.reshape(testx,(testx.shape[0],-1))
target=testx[0,:]

# augmented X trainx:(40, 297, 3)
added=np.sqrt(np.sum(trainx**2,axis=2)) # (40, 297)
trainxAug=np.dstack((trainx,added))  # (40, 297, 4)


# flatten to 2d (40, 297, 3)-->(11880, 3)
trainx_tmp=trainx.transpose(2,0,1)
trainxFlatten=np.reshape(trainx_tmp,(trainx_tmp.shape[0],-1)).T

# optimize: LAG
LAG=100 # ms
shift=int((LAG-50*2)/50) # x already discarded two timepoints during calcuate rate and yank
trainyShift=trainy[:,:(-1)*(shift+2),:] # (40, 299, 180) --> (40, 295, 180)
trainxShift=trainxAug[:,shift:,:] # (40, 297, 4) --> (40, 295, 4)
# flatten training 3D into 2D, concatenate by time dimm (40, 295, 4)-->(4,xxx)  (40, 295, 180)-->(180,xxx)
trainxShift=trainxShift.transpose(2,0,1)
trainxAugFlatten=np.reshape(trainxShift,(trainxShift.shape[0],-1))
trainyShift=trainyShift.transpose(2,0,1)
trainyFlatten=np.reshape(trainyShift,(trainyShift.shape[0],-1))

# fit F, Q
# xnext=F @ xpre + Q
# Optimize: alpha
alpha=20
xpre=np.transpose(trainxFlatten[:-1,:]) # (3, 8909)
xnext=np.transpose(trainxFlatten[1:,:]) # (3, 8909)
rigr1 = Ridge(normalize=False,alpha=alpha)
# xnext.T=xpre.T @ F.T + Q.T
rigr1.fit(xpre.T,xnext.T)
# then F.T=np.transpose(rigr1.coef_) and Q.T=np.transpose(rigr1.intercept_)
# convert to original:xnext=F @ xpre + Q
F=rigr1.coef_ # (3, 3)
fIntercept=rigr1.intercept_ # (3,)
residual1=xnext- F @ xpre - fIntercept[:,np.newaxis] # (3, 8909)
# Optimize : try different denominate
Q=(residual1 @ np.transpose(residual1)) / residual1.shape[1] # (3, 3), 8909 is time length

# fit H, R
# trainyFlatten = H @ trainxAugFlatten + R
rigr2 = Ridge(normalize=False,alpha=alpha)
# trainyFlatten.T = trainxAugFlatten.T @ H.T + R.T
rigr2.fit(trainxAugFlatten.T, trainyFlatten.T)
# rigr2.coef_=H.T.T and rigr2.intercept_=R.T.T
H=rigr2.coef_ # (180, 4)
hIntercept=rigr2.intercept_ # (180,)
residual2=trainyFlatten- H @ trainxAugFlatten - hIntercept[:,np.newaxis]  # (180, 8850)
# Optimize : try different denominate
R=(residual2 @ np.transpose(residual2)) / residual2.shape[1]  # (180, 180)

def f_force(x,dt):
    """ state transition function for a constant velocity """
    fnext = F @ x + fIntercept
    return fnext

def h_forceToNeural(x):
    # construct augmented matrix xaug
    # need at least 3 points to calculate yank.
    #rate=1000*(f[-1][0]-f[-3][0])/(2*dt) # why not use f[-1][1] directly
    #yank=1000*(f[-1][0]-2*f[-2][0]+f[3][0])/(dt**2) # why not use f[-1][2] directly
    another=math.sqrt(x[1]**2 + x[2]**2) # x[1] is rate, x[2] is yank
    xaugment=np.append(x,another)
    return H @ xaugment.T + hIntercept


dt=20 # ms
points = MerweScaledSigmaPoints(n=3, alpha=.1, beta=2., kappa=1.)
ukf = UKF(testx.shape[0], testy.shape[0], dt, fx=f_force, hx=h_forceToNeural, points=points) #(x_dimm, z_dimm,dt, hx, fx, sigmaPoints)
ukf.x = [0.05, 0, 0]  # initial gauss
# Optimize : P
fstd=0.5
ratestd=0.01
yankstd=0.001
P = np.diag([ fstd**2, ratestd**2, yankstd**2])
ukf.P = P
ukf.Q = Q
ukf.R = R

T=int(target.shape[0]/8) # length of one trial
# 74 testing trial in total
startTest=0 # trial to start from
endTest=8 # end with endTest
predictLen=np.arange(startTest*T,endTest*T)
predict, fvar = [], []
for t in predictLen:
    ukf.predict()
    ukf.update(testy[:,t])

    fvar.append(ukf.P[0,0])
    predict.append(ukf.x[0]) #(f,f',f'')

criterion = nn.MSELoss()
mse_loss=criterion(torch.from_numpy(np.asarray(predict)),torch.from_numpy(np.asarray(target)))
figname = plot_dir+"ukfResult"
fig, ax = plt.subplots(2,1,figsize=(12, 6))
predict=np.asarray(predict)
#mse=sum(abs(predict-target[predictLen+LAG-1])**2)
ax[0].text(0.45, 0.9,'MSE: ' + str(np.round(mse_loss.numpy(),4)), fontsize=15,transform=fig.transFigure)
ax[0].plot(target,'r',label='target')
ax[0].plot(predict,'y',label='Kalman prediction')
ax[0].legend()
ax[1].plot(fvar,label='x var')
ax[1].legend()
fig.savefig(figname)
plt.close(fig)
