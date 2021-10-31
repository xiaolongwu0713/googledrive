import numpy as np
from gesture.config import *
import matplotlib.pyplot as plt

sid=10
result_dir=data_dir+'preprocessing'+'/P'+str(sid)+'/' + 'selection/gumbel/'

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

# plot matrix
plt.imshow(z[10][0])





