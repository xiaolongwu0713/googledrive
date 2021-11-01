import numpy as np
from gesture.config import *
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sid=10
#result_dir=data_dir+'preprocessing'+'/P'+str(sid)+'/' + 'selection/gumbel/'
result_dir='/Users/long/OneDrive/share/selection/gumbel/P10'
scores = np.load(result_dir + 'epoch_scores.npy') # (train acc, val acc)
h=np.load(result_dir+'HH.npy')
s=np.load(result_dir+'SS.npy') # selection
z=np.load(result_dir+'ZZ.npy') # probability
h=np.squeeze(h)
z=np.squeeze(z)

mean_entropy=np.mean(h,axis=1)
plt.plot(scores[:,0])
plt.plot(scores[:,1])
plt.plot(mean_entropy)

# best training epoch: how to find the best epoch: lowest entropy + highest val acc
#best_train= np.where(scores == max(scores[:,1]))
best_epoch=70
# plot matrix
plt.imshow(z[2,:,:])

# find the best channel
best_channels=np.argmax(z[best_epoch,:,:],axis=0) # array([148, 153, 148, 153, 149, 152, 153, 152, 152, 149])
best_channels=set(best_channels)

norms=[np.linalg.norm(z[best_epoch,channeli,:], ord=1, axis=0) for channeli in best_channels] # [2.9731846, 2.8495638, 1.7497481, 1.4582942]
[np.linalg.norm(z[6,channeli,:], ord=1, axis=0) for channeli in np.arange(208)]

penalty=norms-thrshold

# penalty

eps = 1e-10
z = torch.clamp(torch.softmax(self.qz_loga, dim=0), eps, 1)  # torch.Size([208, 10])
H = torch.sum(F.relu(torch.norm(z, 1, dim=1) - self.thresh))
# print(max(torch.norm(z, 1, dim=1)-self.thresh)) # penalize
torch.sum(F.relu(torch.norm(z, 1, dim=1)))






